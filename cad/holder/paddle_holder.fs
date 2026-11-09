FeatureScript 3112;
import(path : "onshape/std/common.fs", version : "3112.0");

// SO-101 ping pong paddle holder: replaces Wrist_Roll_Follower_SO101 on the wrist-roll servo horn.
//
// Frame (matches the MuJoCo `gripper` body rotated 180 deg about x, like the stock follower mesh):
//   origin  centre of the servo horn's top face, on the wrist-roll axis
//   +z      along the roll axis, away from the servo, toward the paddle
//   x       blade normal (cap side is +x); the bolts run along x
//   y       across the blade (width); contains the wrist_flex axis direction at roll = 0
//
// Horn interface copied from Wrist_Roll_Follower_SO101.step (TheRobotStudio/SO-ARM100):
//   Ø24 boss 6 mm tall, Ø20.4 x 1 mm horn recess, 4x Ø3.2 on a 9.9 mm square (3 mm under the
//   heads), Ø6 counterbores, Ø5.4 centre hole. Same M3 horn screws as the stock follower. Like the stock part,
//   every hole whose axis lies flat on the print bed is a 45 deg teardrop pointing up (toward the split plane).
//
// The handle butt sits on the pocket floor at z = 9, so paddle x (from butt) = holder z - 9. The pocket follows
// the handle's waist, which locks the paddle axially; the clamp only has to stay closed.
//
// Wrist camera: the stock SO101_Wrist_Cam_Hex-Nut_Mount_32x32_UVC_Module adapter bolts to the -y side, which is
// the robot's left at roll = 0 with the blade level and the cap facing up. It sits where it sits on the stock
// follower (flange face y = -24, screws at z = 23.4, Ø4 hole for its locating peg) with the screw pair moved
// +1.6 mm in x so each half takes one screw. The camera ends up 73 mm out, 6.6 mm behind the horn face, looking
// forward and 25 deg inward; its axis crosses the paddle axis at z ~ 150 (lower blade).
// Nut traps instead of the stock blind pockets: the base nut drops in from the split face, the cap nut from the
// cap's outer face (gravity seats both with the cap up), so they're aligned before the screws go in.
//
// Hardware: the 4 stock M3 horn screws (installed through the pocket with the cap off), 4x M3x10 socket head
// cap screws (M3x8 also works, with ~2.4 mm of thread), 4x M3 heat-set inserts, 4.2 mm OD / 3.8 mm tip, <= 6 mm
// long (e.g. FFVRVSS M3x6) in Ø3.8 holes, pressed flush with the base's split face; camera: 2x M3x8 + 2x M3 nuts.
// Print: each half lying on its flat outer face (layers parallel to the blade, so paddle bending loads run
// along the layers). Only the base's horn boss needs support. PETG or PA-CF, >= 4 perimeters.

// Handle stations [x from butt, width (y), thickness (x)] in mm, from cad/paddle/measurements.yaml
// (caliper station s maps to x = 83 - s; the butt section is held flat for the first 4 mm).
const HANDLE = [[0, 33.88, 24.6], [4, 33.88, 24.6], [23, 30.5, 24.0], [43, 27.18, 23.6], [63, 26.8, 23.5], [83, 29.5, 23.5]];
const HANDLE_CORNER_R = 8; // end-on photo, about +-1 mm

function maxHandleThickness() returns number
{
    var t = 0;
    for (var s in HANDLE)
        t = max(t, s[2]);
    return t;
}

const BOSS_D = 24;
const BOSS_TOP = 5; // above the horn face; the stock follower body starts here too
const RECESS_D = 20.4;
const RECESS_DEPTH = 1;
const HORN_SCREW_D = 3.2;
const HORN_SCREW_HALF_PITCH = 4.95;
const HORN_SCREW_PLATE = 3;
const HORN_HEAD_D = 6;
const HORN_CENTER_D = 5.4;
const BUTT_Z = 9; // pocket floor: the handle butt rests here

const CAP_SCREW_D = 3.4; // M3 clearance
const CAP_HEAD_D = 6.2;
// Cap material under the M3 head. The cap closes up to 0.8 mm onto the handle (0.4 nominal), so an M3x10 reaches
// 4.6-5.0 mm into the insert; an M3x12 would hit the bottom of the 6.5 mm insert hole.
const CAP_UNDER_HEAD = 5;
const BOLT_XS = [24, -10]; // clamp bolt rows, mm along the handle; negative = from the clamp end. The near row
                           // clears the camera screws and nut traps.
const BOLT_EDGE = 4; // bolt axis outboard of the pocket wall

const CAM_FACE_Y = -24; // camera mount flange face
const CAM_SCREW_Z = 23.4;
const CAM_SCREW_X = [-3.4, 4.7]; // stock 8.1 mm pitch; base screw, cap screw
const CAM_SCREW_D = 3.2; // stock follower hole
const CAM_PLATE = 3.5; // flange face to nut (stock): M3x8 under a 2 mm counterbore reaches through the nut
// Nut traps: the first print's 5.7 x 2.7 trap took a 5.48 AF nut in the base but not in the cap, where the trap runs
// up from the bed; 0.4-0.5 mm of slack plus a mouth relief at the cap's bed face (elephant's foot) fix that.
// The open cap trap reaches the bed, so the slicer would grow a support tree up it to the trap's roof:
// make_print_project.py puts a support blocker over it.
const NUT_AF = 5.9; // M3 nut 5.5 across flats + slide clearance; still too narrow for the nut to turn (6.2 across corners)
const NUT_T = 2.9; // M3 nut 2.4 + clearance
const NUT_R = 3.3; // M3 nut half across-corners 3.18 + clearance: the nut lands vertex-first on the trap's end
const NUT_MOUTH = [0.6, 0.4]; // cap trap mouth relief: depth from the bed face, extra per side (not toward the outside)
const CAM_PEG = [-7.1, 13.15]; // [x, z] of the mount's Ø3.4 x 2 locating peg
const CAM_PEG_D = 4;
const CAM_FLANGE_Z = [0, 28.6]; // nothing proud of CAM_FACE_Y here: flange (z 6.05..27.05) + the mount's body behind it
const CAM_PAD_HALF_X = 13; // flat pad filling the sleeve's rounded corners under the flange

const CLEARANCE_BOUNDS = { (millimeter) : [0, 0.2, 2] } as LengthBoundSpec;
const GAP_BOUNDS = { (millimeter) : [0, 0.8, 3] } as LengthBoundSpec;
const FACE_WALL_BOUNDS = { (millimeter) : [1.5, 4, 10] } as LengthBoundSpec;
const SIDE_WALL_BOUNDS = { (millimeter) : [5, 8.5, 15] } as LengthBoundSpec;
const CLAMP_END_BOUNDS = { (millimeter) : [30, 76, 82] } as LengthBoundSpec;
const INSERT_D_BOUNDS = { (millimeter) : [2.5, 3.8, 5.5] } as LengthBoundSpec;
const INSERT_DEPTH_BOUNDS = { (millimeter) : [3, 6.5, 10] } as LengthBoundSpec;

annotation { "Feature Type Name" : "Paddle holder" }
export const paddleHolder = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
        annotation { "Name" : "Clearance per side" }
        isLength(definition.clearance, CLEARANCE_BOUNDS);

        annotation { "Name" : "Split gap (squeeze = gap - 2 x clearance)" }
        isLength(definition.splitGap, GAP_BOUNDS);

        annotation { "Name" : "Wall over blade faces" }
        isLength(definition.faceWall, FACE_WALL_BOUNDS);

        annotation { "Name" : "Wall beside handle" }
        isLength(definition.sideWall, SIDE_WALL_BOUNDS);

        annotation { "Name" : "Clamp end (from handle butt)" }
        isLength(definition.clampEnd, CLAMP_END_BOUNDS);

        annotation { "Name" : "Heat-set insert hole diameter" }
        isLength(definition.insertDiameter, INSERT_D_BOUNDS);

        annotation { "Name" : "Heat-set insert hole depth" }
        isLength(definition.insertDepth, INSERT_DEPTH_BOUNDS);
    }
    {
        const d = {
                "c" : definition.clearance / millimeter,
                "gap" : definition.splitGap / millimeter,
                "faceWall" : definition.faceWall / millimeter,
                "sideWall" : definition.sideWall / millimeter,
                "clampEnd" : definition.clampEnd / millimeter,
                "insertD" : definition.insertDiameter / millimeter,
                "insertDepth" : definition.insertDepth / millimeter
            };

        // Two identical bodies: one becomes the base half (-x side + horn interface), the other the cap (+x side).
        const base = buildBody(context, id + "base", d);
        const cap = buildBody(context, id + "cap", d);

        const capCut = qUnion([
                    fCuboidMm(context, id + "capCutSide", [-100, -100, -100], [d.gap / 2, 100, 300]),
                    fCuboidMm(context, id + "capCutBase", [-100, -100, -100], [100, 100, BUTT_Z])
                ]);
        opBoolean(context, id + "capKeep", {
                    "tools" : capCut,
                    "targets" : cap,
                    "operationType" : BooleanOperationType.SUBTRACTION
                });

        const baseCut = fCuboidMm(context, id + "baseCut", [-d.gap / 2, -100, BUTT_Z], [100, 100, 300]);
        opBoolean(context, id + "baseKeep", {
                    "tools" : baseCut,
                    "targets" : base,
                    "operationType" : BooleanOperationType.SUBTRACTION
                });

        // Clamp bolts along x: counterbored clearance in the cap, heat-set insert holes in the base. The cap prints
        // head-side down, so each counterbore floor is a ceiling: two 0.4 mm steps (a head-wide slot, then a
        // screw-wide square) let it bridge instead of needing support.
        var capHoles = [];
        var baseHoles = [];
        var k = 0;
        for (var row in BOLT_XS)
        {
            const xb = row < 0 ? row + d.clampEnd : row;
            const zb = BUTT_Z + xb;
            const yb = handleAt(xb)[0] / 2 + d.c + BOLT_EDGE;
            for (var side in [-1, 1])
            {
                const y = side * yb;
                capHoles = append(capHoles, fCylinderMm(context, id + ("capHole" ~ k), [0, y, zb], [100, y, zb], CAP_SCREW_D));
                const cb = d.gap / 2 + CAP_UNDER_HEAD;
                const hs = CAP_SCREW_D / 2;
                capHoles = append(capHoles, fCylinderMm(context, id + ("capHead" ~ k), [cb, y, zb], [100, y, zb], CAP_HEAD_D));
                capHoles = append(capHoles, fCuboidMm(context, id + ("capBridge1" ~ k), [cb - 0.4, y - CAP_HEAD_D / 2, zb - hs], [cb, y + CAP_HEAD_D / 2, zb + hs]));
                capHoles = append(capHoles, fCuboidMm(context, id + ("capBridge2" ~ k), [cb - 0.8, y - hs, zb - hs], [cb - 0.4, y + hs, zb + hs]));
                baseHoles = append(baseHoles, fCylinderMm(context, id + ("insert" ~ k), [-d.gap / 2 - d.insertDepth, y, zb], [0, y, zb], d.insertD));
                k += 1;
            }
        }
        opBoolean(context, id + "capBolts", { "tools" : qUnion(capHoles), "targets" : cap, "operationType" : BooleanOperationType.SUBTRACTION });
        opBoolean(context, id + "baseBolts", { "tools" : qUnion(baseHoles), "targets" : base, "operationType" : BooleanOperationType.SUBTRACTION });

        setProperty(context, { "entities" : base, "propertyType" : PropertyType.NAME, "value" : "Paddle holder base" });
        setProperty(context, { "entities" : cap, "propertyType" : PropertyType.NAME, "value" : "Paddle holder cap" });
        println("base volume cm3: " ~ roundToPrecision(evVolume(context, { "entities" : base }) / (centimeter ^ 3), 2));
        println("cap volume cm3: " ~ roundToPrecision(evVolume(context, { "entities" : cap }) / (centimeter ^ 3), 2));
        println("base bbox: " ~ toString(evBox3d(context, { "topology" : base })));
        println("cap bbox: " ~ toString(evBox3d(context, { "topology" : cap })));
    }, {
            clearance : 0.2 * millimeter,
            splitGap : 0.8 * millimeter,
            faceWall : 4 * millimeter,
            sideWall : 8.5 * millimeter,
            clampEnd : 76 * millimeter,
            insertDiameter : 3.8 * millimeter,
            insertDepth : 6.5 * millimeter
        });

/** Full clamp (sleeve around the handle + horn boss + camera pad) with every hole cut; returns the solid. */
function buildBody(context is Context, id is Id, d is map) returns Query
{
    // Outer sleeve: offset handle sections, starting at the top of the boss. The faces over the blade are flat
    // (thickest handle section) so each half prints lying on its outer face.
    const sleeve = loftSections(context, id + "sleeve", sectionXs(BOSS_TOP - BUTT_Z, d.clampEnd),
        d.c + d.sideWall, d.c + d.faceWall, true);
    const boss = fCylinderMm(context, id + "boss", [0, 0, -RECESS_DEPTH], [0, 0, BOSS_TOP + 0.5], BOSS_D);
    // Faces over the blade: each half's print bed. What overhangs them is cut at 45 deg, except the boss: the wrist
    // bracket comes to r = 12.6 just behind the horn face and r = 15 up to z = 4.3, which leaves no room for a keel
    // down to the bed, so the boss gets a small tree support.
    const hx = d.c + d.faceWall + maxHandleThickness() / 2;
    // Camera pad: flat seat at y = CAM_FACE_Y, 45 deg chamfers down to both beds.
    const p = CAM_PAD_HALF_X;
    const f = CAM_FACE_Y;
    const camPad = fPrismMm(context, id + "camPad", [[-p, f], [p, f], [hx, f + hx - p], [hx, f + 10], [-hx, f + 10], [-hx, f + hx - p]], BOSS_TOP, CAM_FLANGE_Z[1]);
    opBoolean(context, id + "join", { "tools" : qUnion([sleeve, boss, camPad]), "operationType" : BooleanOperationType.UNION });

    // Handle pocket, running past the end of the sleeve so it opens toward the blade.
    var cutters = [loftSections(context, id + "pocket", sectionXs(0, d.clampEnd + 4), d.c, d.c, false)];
    cutters = append(cutters, fCylinderMm(context, id + "recess", [0, 0, -RECESS_DEPTH - 1], [0, 0, 0], RECESS_D));

    // Horn holes run along z, flat on the bed: the base prints with +x up.
    const up = [1, 0, 0];
    cutters = append(cutters, fTeardropMm(context, id + "center", [0, 0, -1], [0, 0, BUTT_Z + 0.1], HORN_CENTER_D, up));
    var k = 0;
    for (var sx in [-1, 1])
    {
        for (var sy in [-1, 1])
        {
            const x = sx * HORN_SCREW_HALF_PITCH;
            const y = sy * HORN_SCREW_HALF_PITCH;
            cutters = append(cutters, fTeardropMm(context, id + ("hornHole" ~ k), [x, y, -1], [x, y, HORN_SCREW_PLATE + 0.1], HORN_SCREW_D, up));
            cutters = append(cutters, fTeardropMm(context, id + ("hornHead" ~ k), [x, y, HORN_SCREW_PLATE], [x, y, BUTT_Z + 0.1], HORN_HEAD_D, up));
            k += 1;
        }
    }

    // Camera mount: flat flange seat, locating-peg hole (base), one screw + nut trap per half. Each half prints
    // with its split face up, so teardrops point toward x = 0.
    cutters = append(cutters, fCuboidMm(context, id + "camSeat", [-100, CAM_FACE_Y - 100, CAM_FLANGE_Z[0]], [100, CAM_FACE_Y, CAM_FLANGE_Z[1]]));
    cutters = append(cutters, fTeardropMm(context, id + "camPeg", [CAM_PEG[0], CAM_FACE_Y - 1, CAM_PEG[1]], [CAM_PEG[0], CAM_FACE_Y + 2.6, CAM_PEG[1]], CAM_PEG_D, up));
    const nutY = [CAM_FACE_Y + CAM_PLATE, CAM_FACE_Y + CAM_PLATE + NUT_T];
    const nutZ = [CAM_SCREW_Z - NUT_AF / 2, CAM_SCREW_Z + NUT_AF / 2];
    for (var x in CAM_SCREW_X)
    {
        const sx = x < 0 ? -1 : 1;
        cutters = append(cutters, fTeardropMm(context, id + ("camScrew" ~ sx), [x, CAM_FACE_Y - 1, CAM_SCREW_Z], [x, nutY[1] + 0.8, CAM_SCREW_Z], CAM_SCREW_D, [-sx, 0, 0]));
        // Base trap opens on the split face; cap trap opens on the cap's outer face, which is its print bed.
        const trapX = sx < 0 ? [x - NUT_R, 0] : [x - NUT_R, 100];
        cutters = append(cutters, fCuboidMm(context, id + ("camNut" ~ sx), [trapX[0], nutY[0], nutZ[0]], [trapX[1], nutY[1], nutZ[1]]));
        if (sx > 0)
        {
            const m = NUT_MOUTH[1];
            cutters = append(cutters, fCuboidMm(context, id + "camNutMouth", [hx - NUT_MOUTH[0], nutY[0], nutZ[0] - m], [100, nutY[1] + m, nutZ[1] + m]));
        }
    }

    const body = qSubtraction(qBodyType(qCreatedBy(id, EntityType.BODY), BodyType.SOLID), qUnion(cutters));
    opBoolean(context, id + "cut", { "tools" : qUnion(cutters), "targets" : body, "operationType" : BooleanOperationType.SUBTRACTION });
    return qBodyType(qCreatedBy(id, EntityType.BODY), BodyType.SOLID);
}

/**
 * Cylinder plus a 45 deg point toward `tip` (unit vector perpendicular to the axis). With `tip` pointing up, a
 * hole lying flat on the bed prints without support.
 */
function fTeardropMm(context is Context, id is Id, bottom is array, top is array, diameter is number, tip is array) returns Query
{
    const cylinder = fCylinderMm(context, id + "cyl", bottom, top, diameter);
    const axis = vector(top) - vector(bottom);
    const len = norm(axis);
    const s = diameter / 2 / sqrt(2);
    const sid = id + "sk";
    const sketch = newSketchOnPlane(context, sid, { "sketchPlane" : plane(vector(bottom) * millimeter, axis / len, vector(tip)) });
    // Square with one corner on the axis and the opposite corner at the tip: its sides are tangent to the circle.
    skPolyline(sketch, "point", { "points" : [vector(0, 0) * millimeter, vector(s, -s) * millimeter, vector(2 * s, 0) * millimeter,
                    vector(s, s) * millimeter, vector(0, 0) * millimeter] });
    skSolve(sketch);
    opExtrude(context, id + "point", {
                "entities" : qSketchRegion(sid),
                "direction" : axis / len,
                "endBound" : BoundingType.BLIND,
                "endDepth" : len * millimeter
            });
    opDeleteBodies(context, id + "deleteSketch", { "entities" : qCreatedBy(sid, EntityType.BODY) });
    return qUnion([cylinder, qCreatedBy(id + "point", EntityType.BODY)]);
}

/** Prism over a closed x-y polygon (mm, no repeated end point), from z0 to z1. */
function fPrismMm(context is Context, id is Id, points is array, z0 is number, z1 is number) returns Query
{
    const sid = id + "sk";
    const sketch = newSketchOnPlane(context, sid, { "sketchPlane" : plane(vector(0, 0, z0) * millimeter, vector(0, 0, 1), vector(1, 0, 0)) });
    var pts = [];
    for (var pt in append(points, points[0]))
        pts = append(pts, vector(pt) * millimeter);
    skPolyline(sketch, "outline", { "points" : pts });
    skSolve(sketch);
    opExtrude(context, id + "extrude", {
                "entities" : qSketchRegion(sid),
                "direction" : vector(0, 0, 1),
                "endBound" : BoundingType.BLIND,
                "endDepth" : (z1 - z0) * millimeter
            });
    opDeleteBodies(context, id + "deleteSketch", { "entities" : qCreatedBy(sid, EntityType.BODY) });
    return qCreatedBy(id + "extrude", EntityType.BODY);
}

/**
 * Lofted solid through handle sections at the given x (mm from butt), offset outward by offsetWidth (y) and
 * offsetThickness (x). flatFaces holds the thickness at the handle's maximum and uses printableRect sections.
 */
function loftSections(context is Context, id is Id, xs is array, offsetWidth is number, offsetThickness is number, flatFaces is boolean) returns Query
{
    var profiles = [];
    for (var i = 0; i < size(xs); i += 1)
    {
        const wt = handleAt(xs[i]);
        const sid = id + "sk" + ("s" ~ i);
        const sketch = newSketchOnPlane(context, sid, {
                    "sketchPlane" : plane(vector(0, 0, BUTT_Z + xs[i]) * millimeter, vector(0, 0, 1), vector(1, 0, 0))
                });
        // Sketch x = world x (thickness), sketch y = world y (width).
        const t = flatFaces ? maxHandleThickness() : wt[1];
        const r = HANDLE_CORNER_R + min(offsetWidth, offsetThickness);
        if (flatFaces)
            printableRect(sketch, t + 2 * offsetThickness, wt[0] + 2 * offsetWidth, r);
        else
            roundedRect(sketch, t + 2 * offsetThickness, wt[0] + 2 * offsetWidth, r);
        skSolve(sketch);
        profiles = append(profiles, qSketchRegion(sid));
    }
    opLoft(context, id + "loft", { "profileSubqueries" : profiles, "bodyType" : ToolBodyType.SOLID });
    opDeleteBodies(context, id + "deleteSketches", { "entities" : qCreatedBy(id + "sk", EntityType.BODY) });
    return qCreatedBy(id + "loft", EntityType.BODY);
}

/** Closed rounded rectangle centred on the sketch origin; sizes in mm. */
function roundedRect(sketch is Sketch, sizeX is number, sizeY is number, radius is number)
{
    const hx = sizeX / 2;
    const hy = sizeY / 2;
    const r = min(radius, min(hx, hy) - 0.5);
    const q = r / sqrt(2);
    const mm = millimeter;
    skLineSegment(sketch, "bottom", { "start" : vector(-hx + r, -hy) * mm, "end" : vector(hx - r, -hy) * mm });
    skArc(sketch, "br", { "start" : vector(hx - r, -hy) * mm, "mid" : vector(hx - r + q, -hy + r - q) * mm, "end" : vector(hx, -hy + r) * mm });
    skLineSegment(sketch, "right", { "start" : vector(hx, -hy + r) * mm, "end" : vector(hx, hy - r) * mm });
    skArc(sketch, "tr", { "start" : vector(hx, hy - r) * mm, "mid" : vector(hx - r + q, hy - r + q) * mm, "end" : vector(hx - r, hy) * mm });
    skLineSegment(sketch, "top", { "start" : vector(hx - r, hy) * mm, "end" : vector(-hx + r, hy) * mm });
    skArc(sketch, "tl", { "start" : vector(-hx + r, hy) * mm, "mid" : vector(-hx + r - q, hy - r + q) * mm, "end" : vector(-hx, hy - r) * mm });
    skLineSegment(sketch, "left", { "start" : vector(-hx, hy - r) * mm, "end" : vector(-hx, -hy + r) * mm });
    skArc(sketch, "bl", { "start" : vector(-hx, -hy + r) * mm, "mid" : vector(-hx + r - q, -hy + r - q) * mm, "end" : vector(-hx + r, -hy) * mm });
}

/**
 * Rounded rectangle whose corners turn into 45 deg chamfers where they meet the x faces, so a half lying on either
 * x face prints its edges without support. Stays inside the plain rounded rectangle's circumscribed radius.
 */
function printableRect(sketch is Sketch, sizeX is number, sizeY is number, radius is number)
{
    const hx = sizeX / 2;
    const hy = sizeY / 2;
    const r = min(radius, min(hx, hy) - 0.5);
    const q = r / sqrt(2); // arc stops at 45 deg
    const c = r - q; // ...this far in from each face
    const e = r * (sqrt(2) - 1); // chamfer meets the x face this far beyond the arc's tangent point
    const m = r * cos(22.5 * degree); // arc midpoints at 67.5 deg from the x faces
    const n = r * sin(22.5 * degree);
    const mm = millimeter;
    skLineSegment(sketch, "right", { "start" : vector(hx, -hy + r - e) * mm, "end" : vector(hx, hy - r + e) * mm });
    skLineSegment(sketch, "trC", { "start" : vector(hx, hy - r + e) * mm, "end" : vector(hx - c, hy - c) * mm });
    skArc(sketch, "tr", { "start" : vector(hx - c, hy - c) * mm, "mid" : vector(hx - r + n, hy - r + m) * mm, "end" : vector(hx - r, hy) * mm });
    skLineSegment(sketch, "top", { "start" : vector(hx - r, hy) * mm, "end" : vector(-hx + r, hy) * mm });
    skArc(sketch, "tl", { "start" : vector(-hx + r, hy) * mm, "mid" : vector(-hx + r - n, hy - r + m) * mm, "end" : vector(-hx + c, hy - c) * mm });
    skLineSegment(sketch, "tlC", { "start" : vector(-hx + c, hy - c) * mm, "end" : vector(-hx, hy - r + e) * mm });
    skLineSegment(sketch, "left", { "start" : vector(-hx, hy - r + e) * mm, "end" : vector(-hx, -hy + r - e) * mm });
    skLineSegment(sketch, "blC", { "start" : vector(-hx, -hy + r - e) * mm, "end" : vector(-hx + c, -hy + c) * mm });
    skArc(sketch, "bl", { "start" : vector(-hx + c, -hy + c) * mm, "mid" : vector(-hx + r - n, -hy + r - m) * mm, "end" : vector(-hx + r, -hy) * mm });
    skLineSegment(sketch, "bottom", { "start" : vector(-hx + r, -hy) * mm, "end" : vector(hx - r, -hy) * mm });
    skArc(sketch, "br", { "start" : vector(hx - r, -hy) * mm, "mid" : vector(hx - r + n, -hy + r - m) * mm, "end" : vector(hx - c, -hy + c) * mm });
    skLineSegment(sketch, "brC", { "start" : vector(hx - c, -hy + c) * mm, "end" : vector(hx, -hy + r - e) * mm });
}

/** Section x positions: xStart, every handle station strictly inside, xEnd. */
function sectionXs(xStart is number, xEnd is number) returns array
{
    var xs = [xStart];
    for (var s in HANDLE)
    {
        if (s[0] > xStart + 0.5 && s[0] < xEnd - 0.5)
            xs = append(xs, s[0]);
    }
    return append(xs, xEnd);
}

/** [width, thickness] of the handle at x (mm from butt), linearly interpolated, held flat past the ends. */
function handleAt(x is number) returns array
{
    if (x <= HANDLE[0][0])
        return [HANDLE[0][1], HANDLE[0][2]];
    for (var i = 1; i < size(HANDLE); i += 1)
    {
        if (x <= HANDLE[i][0])
        {
            const a = HANDLE[i - 1];
            const b = HANDLE[i];
            const f = (x - a[0]) / (b[0] - a[0]);
            return [a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])];
        }
    }
    const last = HANDLE[size(HANDLE) - 1];
    return [last[1], last[2]];
}

function fCylinderMm(context is Context, id is Id, bottom is array, top is array, diameter is number) returns Query
{
    fCylinder(context, id, {
                "bottomCenter" : vector(bottom) * millimeter,
                "topCenter" : vector(top) * millimeter,
                "radius" : diameter / 2 * millimeter
            });
    return qCreatedBy(id, EntityType.BODY);
}

function fCuboidMm(context is Context, id is Id, corner1 is array, corner2 is array) returns Query
{
    fCuboid(context, id, { "corner1" : vector(corner1) * millimeter, "corner2" : vector(corner2) * millimeter });
    return qCreatedBy(id, EntityType.BODY);
}
