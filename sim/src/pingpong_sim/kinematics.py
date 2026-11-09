"""Batched forward/inverse kinematics of the arm on torch (GPU), for controllers running in many worlds at once.

The chain is read from a compiled MuJoCo model (robot entity, any prefix): base -> shoulder -> upper_arm ->
lower_arm -> wrist -> gripper -> paddle_holder -> paddle, one servo joint per link. Joint angles are encoder angles:
the gear joint shares the servo joint's axis and origin, so their sum is a single rotation.

Output frames: the cap-side blade face site (`blade_face_cap`): position of the face centre and its outward normal
(the site z axis), in the world frame; and the wrist camera (`wrist_cam`), in the OpenCV convention the ball filter
uses (x right, y down, z along the optical axis).
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from pingpong_sim.robot import ARM_JOINTS

CHAIN = ("shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "paddle_holder", "paddle")
JOINT_OF = dict(zip(CHAIN, ARM_JOINTS))  # paddle_holder and paddle carry no joint


def _quat_mat(q: np.ndarray) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


class ArmKinematics:
    def __init__(self, mj_model: mujoco.MjModel, prefix: str = "robot/", device: str = "cpu"):
        d = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, d)
        base = mj_model.body(prefix + "base").id
        f = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=device)  # noqa: E731
        self.device = device
        self.base_R, self.base_p = f(d.xmat[base].reshape(3, 3)), f(d.xpos[base])
        self.links = []  # (R_local, p_local, joint axis or None, joint pos)
        for name in CHAIN:
            b = mj_model.body(prefix + name)
            axis = jpos = None
            if name in JOINT_OF:
                j = mj_model.joint(prefix + JOINT_OF[name])
                axis, jpos = f(j.axis), f(j.pos)
            self.links.append((f(_quat_mat(b.quat)), f(b.pos), axis, jpos))
        s = mj_model.site(prefix + "blade_face_cap")
        self.site_R, self.site_p = f(_quat_mat(s.quat)), f(s.pos)
        # Wrist camera relative to the last chain link, from the posed default model; MuJoCo cameras look along -z
        # with +y up, so flip y and z for OpenCV.
        last = mj_model.body(prefix + CHAIN[-1]).id
        cam = mj_model.camera(prefix + "wrist_cam").id
        R_l, R_c = d.xmat[last].reshape(3, 3), d.cam_xmat[cam].reshape(3, 3)
        self.cam_R = f(R_l.T @ R_c @ np.diag([1.0, -1.0, -1.0]))
        self.cam_p = f(R_l.T @ (d.cam_xpos[cam] - d.xpos[last]))
        jid = [mj_model.joint(prefix + j).id for j in ARM_JOINTS]
        self.lo, self.hi = f(mj_model.jnt_range[jid, 0]), f(mj_model.jnt_range[jid, 1])

    @staticmethod
    def _rot(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """Rodrigues: (B,) angles about a fixed unit axis -> (B, 3, 3)."""
        k = axis / axis.norm()
        K = torch.zeros(3, 3, device=axis.device)
        K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
        s, c = torch.sin(angle)[:, None, None], torch.cos(angle)[:, None, None]
        return torch.eye(3, device=axis.device) + s * K + (1 - c) * (K @ K)

    def _chain(self, q: torch.Tensor):
        """Last link's world rotation (B, 3, 3) and position (B, 3), joint origins and world joint axes (B, 5, 3)."""
        B = q.shape[0]
        R = self.base_R.expand(B, 3, 3)
        p = self.base_p.expand(B, 3)
        origins, axes = [], []
        k = 0
        for R_l, p_l, axis, jpos in self.links:
            p = p + (R @ p_l)
            R = R @ R_l
            if axis is not None:
                o = p + R @ jpos
                a = R @ axis
                origins.append(o)
                axes.append(a)
                Rj = self._rot(axis, q[:, k])
                # Rotation about the joint origin: p stays where it is if jpos = 0 (true for this model).
                p = o - (R @ (Rj @ jpos.expand(B, 3).unsqueeze(-1))).squeeze(-1)
                R = R @ Rj
                k += 1
        return R, p, torch.stack(origins, 1), torch.stack(axes, 1)

    def fk(self, q: torch.Tensor):
        """q (B, 5) encoder angles -> face centre (B, 3), face normal (B, 3), joint origins (B, 5, 3) and world
        joint axes (B, 5, 3)."""
        R, p, origins, axes = self._chain(q)
        face = p + R @ self.site_p
        normal = (R @ self.site_R)[:, :, 2]
        return face, normal, origins, axes

    def wrist_camera(self, q: torch.Tensor):
        """q (B, 5) encoder angles -> wrist camera world-from-camera rotation (B, 3, 3; OpenCV axes) and optical
        centre (B, 3)."""
        R, p, _, _ = self._chain(q)
        return R @ self.cam_R, p + R @ self.cam_p

    def jacobian(self, face: torch.Tensor, normal: torch.Tensor, origins: torch.Tensor, axes: torch.Tensor):
        """d(face)/dq and d(normal)/dq, each (B, 3, 5), for revolute joints."""
        Jp = torch.cross(axes, face[:, None, :] - origins, dim=-1).transpose(1, 2)
        Jn = torch.cross(axes, normal[:, None, :].expand_as(axes), dim=-1).transpose(1, 2)
        return Jp, Jn

    def ik(self, face_target: torch.Tensor, normal_target: torch.Tensor, q0: torch.Tensor, iters: int = 8,
           normal_weight: float = 0.3, damping: float = 1e-4):
        """Damped Gauss-Newton from q0 (B, 5): face centre at face_target, face normal along normal_target.
        Returns q (B, 5) within the joint limits and the residual position error [m] (B,)."""
        q = q0.clone()
        eye = torch.eye(5, device=q.device)
        for _ in range(iters):
            face, normal, o, a = self.fk(q)
            Jp, Jn = self.jacobian(face, normal, o, a)
            J = torch.cat([Jp, normal_weight * Jn], dim=1)  # (B, 6, 5)
            r = torch.cat([face - face_target, normal_weight * (normal - normal_target)], dim=1)
            JT = J.transpose(1, 2)
            dq = torch.linalg.solve(JT @ J + damping * eye, JT @ r.unsqueeze(-1)).squeeze(-1)
            q = torch.clamp(q - dq, self.lo, self.hi)
        face, _, _, _ = self.fk(q)
        return q, (face - face_target).norm(dim=-1)
