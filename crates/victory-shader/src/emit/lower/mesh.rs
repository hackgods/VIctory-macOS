//! Mesh shader output signatures, counts and lane indices.

use super::{Attr, HashMap, Lower, SigElement, V, Why, pred};
use crate::emit::shader::Mesh;

/// Semantic key `(UPPERCASE_NAME, index)` -> 4-bit absolute channel mask of
/// arbitrary (`kind == 0`) inputs read by the paired pixel shader.
pub(crate) type KeptOutputs = HashMap<(String, u32), u8>;

/// Keeps only arbitrary (`kind == 0`) signature elements whose semantic has at
/// least one channel read in `keep`, trimming unread leading and trailing
/// channels. System values (`kind != 0`) stay untouched.
pub(super) fn prune_signature(elements: &mut Vec<SigElement>, keep: &KeptOutputs) {
    elements.retain_mut(|e| {
        if e.kind != 0 {
            return true;
        }
        if e.row.is_none() {
            return false;
        }
        let key = (e.name.to_ascii_uppercase(), e.index);
        let used = keep.get(&key).copied().unwrap_or(0);
        let rel = (used >> e.col) & ((1u8 << e.cols) - 1);
        if rel == 0 {
            return false;
        }
        let first = rel.trailing_zeros() as u8;
        let last = 7 - rel.leading_zeros() as u8;
        e.col += first;
        e.cols = last - first + 1;
        true
    });
}

/// The reserved [`KeptOutputs`] key under which a linked translation carries
/// the rows to merge (no semantic has this name); the value is the mode.
const MERGE_KEY: &str = "\u{0}merge-row";
/// The reserved [`KeptOutputs`] key under which a linked translation carries
/// the host register of a console row ([`plan_merge`]); the value is the
/// register.
const ORDER_KEY: &str = "\u{0}host-row";
/// The semantic of a merged row; its index is the row.
const MERGED_NAME: &str = "VROW";

/// macOS port (Metal's 32 KiB mesh output limit): marks in `keep` the rows
/// of `outputs` that hold two or more arbitrary elements once pruned, all of
/// one component type, so both stages of a linked pair make each such row
/// one element ([`merge_rows`]). D3DMetal sizes a row of several elements
/// (`xyz` and `w`, say) well above the 16 bytes one `float4` takes.
///
/// With `reorder` the arbitrary rows also trade registers among themselves so
/// that whole `float4` rows come first, then three, two and one components:
/// Metal lays the rows out in register order, a `float4` on a 16-byte
/// boundary and three components packed in 12 bytes, so a `float4` after an
/// odd number of `xyz` rows costs 4 bytes of padding (notes/mesh-size.md).
///
/// With `distances_last` as well, rows that hold only clip or cull distances
/// take the last registers, after the arbitrary rows. This alone saves
/// nothing: D3DMetal declares the distances twice, as Metal's clip-distance
/// array right after the position (one 4-byte slot, wherever their register
/// is) and as an ordinary output of their width at their register.
///
/// With `xyz_first` as well, and only where such rows exist, the arbitrary
/// rows go three components first, then two and one, the whole `float4`
/// rows last: after that 4-byte slot `float4` rows cost Metal padding that
/// rows of three do not (measured with Metal itself, notes/mesh-size.md,
/// "Correction, 26 November afternoon").
pub(crate) fn plan_merge(
    keep: &mut KeptOutputs,
    outputs: &[SigElement],
    widen: bool,
    reorder: bool,
    distances_last: bool,
    xyz_first: bool,
) {
    let mut pruned = outputs.to_vec();
    prune_signature(&mut pruned, keep);
    let arbitrary = |r: u32| pruned.iter().filter(move |e| e.kind == 0 && e.row == Some(r));
    let rows: std::collections::BTreeSet<u32> =
        pruned.iter().filter(|e| e.kind == 0).filter_map(|e| e.row).collect();
    for r in rows {
        let first = arbitrary(r).next().expect("a row from the list");
        let same = arbitrary(r).all(|e| e.comp == first.comp && e.half == first.half);
        if same && (widen || arbitrary(r).count() > 1) {
            keep.insert((MERGE_KEY.to_string(), r), 1 + u8::from(widen));
        }
    }
    if reorder {
        merge_rows(&mut pruned, keep);
        // Rows still shared by elements of several types keep their place.
        let mut rows: Vec<(u32, u8)> = Vec::new();
        for e in pruned.iter().filter(|e| e.kind == 0) {
            let Some(r) = e.row else { continue };
            match rows.iter_mut().find(|(row, _)| *row == r) {
                Some((_, cols)) => *cols = 0,
                None => rows.push((r, e.cols)),
            }
        }
        rows.retain(|&(_, cols)| cols != 0);
        let mut registers: Vec<u32> = rows.iter().map(|&(r, _)| r).collect();
        // Rows of distances alone, in their order (pos1 and pos2 find them
        // by kind and index, not by row).
        let distance = |e: &SigElement| matches!(e.kind, 6 | 7);
        let mut dists: Vec<u32> = Vec::new();
        if distances_last {
            dists = pruned.iter().filter(|e| distance(e)).filter_map(|e| e.row).collect();
            dists.sort_unstable();
            dists.dedup();
            dists.retain(|&r| pruned.iter().all(|e| e.row != Some(r) || distance(e)));
        }
        if xyz_first && !dists.is_empty() {
            // 3, 2, 1, then 4 components.
            rows.sort_by_key(|&(r, cols)| (if cols == 4 { 3 } else { 3 - cols }, r));
        } else {
            rows.sort_by_key(|&(r, cols)| (std::cmp::Reverse(cols), r));
        }
        registers.extend(&dists);
        rows.extend(dists.iter().map(|&r| (r, 0)));
        registers.sort_unstable();
        for (&(r, _), register) in rows.iter().zip(registers) {
            keep.insert((ORDER_KEY.to_string(), r), register as u8);
        }
    }
}

/// The host register [`plan_merge`] gave the arbitrary (or distance) element
/// at console row `row`, if it moved it.
fn host_row(keep: &KeptOutputs, row: u32) -> Option<u32> {
    keep.get(&(ORDER_KEY.to_string(), row)).map(|&r| u32::from(r))
}

/// The elements [`plan_merge`] may give other registers: arbitrary ones, and
/// clip and cull distances.
fn movable(e: &SigElement) -> bool {
    matches!(e.kind, 0 | 6 | 7)
}

/// Gives the elements the registers [`plan_merge`] chose. Last: lowering
/// finds an element by its console row.
pub(super) fn renumber_rows(elements: &mut [SigElement], keep: &KeptOutputs) {
    for e in elements.iter_mut().filter(|e| movable(e)) {
        if let Some(register) = e.row.and_then(|r| host_row(keep, r)) {
            e.row = Some(register);
        }
    }
}

/// Makes each row [`plan_merge`] marked one element from column 0 to the
/// row's last used column (all four when the plan widens), named by its row,
/// so the mesh and the pixel shader of a pair declare the same thing whatever
/// either leaves out. Exports and reads find their channel by row and
/// column as before.
pub(super) fn merge_rows(elements: &mut Vec<SigElement>, keep: &KeptOutputs) {
    let mut done = std::collections::BTreeSet::new();
    let mut at = 0;
    while at < elements.len() {
        let e = &elements[at];
        let mode = match (e.kind, e.row) {
            (0, Some(r)) => keep.get(&(MERGE_KEY.to_string(), r)).copied().map(|m| (r, m)),
            _ => None,
        };
        let Some((r, mode)) = mode else {
            at += 1;
            continue;
        };
        if !done.insert(r) {
            // A later element of a merged row: the first one holds it.
            elements.remove(at);
            continue;
        }
        let (mut end, mut usage) = (0, 0);
        for o in elements.iter().filter(|o| o.kind == 0 && o.row == Some(r)) {
            end = end.max(o.col + o.cols);
            usage |= o.usage << o.col;
        }
        let e = &mut elements[at];
        e.name = MERGED_NAME.to_string();
        e.index = r;
        e.col = 0;
        e.cols = if mode == 2 { 4 } else { end };
        e.usage = usage;
        at += 1;
    }
    // The signature lists the elements that move in the order of their host
    // registers, in the places those elements had.
    let slots: Vec<usize> = (0..elements.len())
        .filter(|&i| movable(&elements[i]) && elements[i].row.is_some_and(|r| host_row(keep, r).is_some()))
        .collect();
    let mut moved: Vec<SigElement> = slots.iter().map(|&i| elements[i].clone()).collect();
    moved.sort_by_key(|e| e.row.and_then(|r| host_row(keep, r)));
    for (i, e) in slots.into_iter().zip(moved) {
        elements[i] = e;
    }
}

impl Lower<'_> {
    /// Mesh: the counts and the primitives no lane exports. The console's
    /// counts come from one lane's allocation request, which DXIL cannot
    /// express (one call, dominating every output write).
    // shortcut: the counts are the declared maxima and each lane first culls
    // its primitive (all indices 0, zero area), so primitives past the
    // program's count are culled and vertices past it unused; this costs
    // the host the unused vertex and primitive slots. Pass the program's own
    // counts if that shows in frame time.
    pub(super) fn mesh_prologue(&mut self, mesh: Mesh) -> Result<(), Why> {
        let threads = self.s.threads.expect("mesh")[0];
        if mesh.topology != 2 {
            return Err("a mesh shader not drawing triangles");
        }
        if threads > 128 || mesh.max_vertices > 256 || mesh.max_primitives > 256 {
            return Err("a mesh shader beyond the host's limits");
        }
        let e = self.s.entry;
        let void = self.s.t.void;
        let (v, p) = (
            self.s.m.i32(mesh.max_vertices),
            self.s.m.i32(mesh.max_primitives),
        );
        self.s
            .op(e, "setMeshOutputCounts", Attr::NoUnwind, void, 168, &[v, p]);
        let flat = self.mesh_thread();
        let guard = (threads > mesh.max_primitives).then(|| self.s.m.cmp(e, pred::IULT, flat, p));
        let at = self.begin_guarded(guard);
        let z = self.s.m.i32(0);
        self.s.op(
            self.cur,
            "emitIndices",
            Attr::NoUnwind,
            void,
            169,
            &[flat, z, z, z],
        );
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// Mesh: the thread's index in the group, which is its vertex and
    /// primitive index.
    pub(super) fn mesh_thread(&mut self) -> V {
        if let Some(f) = self.flat {
            return f;
        }
        let e = self.s.entry;
        let f = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            self.s.t.i32,
            96,
            &[],
        );
        self.flat = Some(f);
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn element(name: &str, index: u32, kind: u8, comp: u8, row: u32, col: u8, cols: u8) -> SigElement {
        SigElement {
            name: name.to_string(),
            index,
            kind,
            comp,
            interp: 0,
            row: Some(row),
            col,
            cols,
            usage: (1 << cols) - 1,
            half: false,
        }
    }

    /// A row shared by `xyz` and `w` elements becomes one `float4` named by
    /// the row in both stages, whichever of the two a stage declares first;
    /// rows of one element, of mixed types and system values stay.
    #[test]
    fn rows_of_several_elements_merge_the_same_way_in_both_stages() {
        let outputs = vec![
            element("SV_Position", 0, 3, 9, 0, 0, 4),
            element("TEXCOORD", 4, 0, 9, 5, 0, 3),
            element("TEXCOORD", 7, 0, 9, 5, 3, 1),
            element("TEXCOORD", 5, 0, 9, 6, 0, 3),
            element("COLOR", 3, 0, 5, 7, 0, 1),
            element("TEXCOORD", 20, 0, 9, 7, 1, 1),
        ];
        let mut keep = KeptOutputs::new();
        for e in outputs.iter().filter(|e| e.kind == 0) {
            keep.insert((e.name.clone(), e.index), 0xF);
        }
        plan_merge(&mut keep, &outputs, false, false, false, false);
        let mut mesh = outputs.clone();
        merge_rows(&mut mesh, &keep);
        let shape = |els: &[SigElement]| -> Vec<(String, u32, Option<u32>, u8, u8)> {
            els.iter().map(|e| (e.name.clone(), e.index, e.row, e.col, e.cols)).collect()
        };
        assert_eq!(
            shape(&mesh),
            vec![
                ("SV_Position".to_string(), 0, Some(0), 0, 4),
                ("VROW".to_string(), 5, Some(5), 0, 4),
                ("TEXCOORD".to_string(), 5, Some(6), 0, 3),
                ("COLOR".to_string(), 3, Some(7), 0, 1),
                ("TEXCOORD".to_string(), 20, Some(7), 1, 1),
            ]
        );
        assert_eq!(mesh[1].usage, 0xF);
        // The pixel shader declares the `w` element first and no position.
        let mut pixel = vec![outputs[2].clone(), outputs[1].clone(), outputs[3].clone()];
        merge_rows(&mut pixel, &keep);
        assert_eq!(shape(&pixel), shape(&mesh[1..3]));
        // Widening makes every arbitrary row of one type a whole float4.
        let mut keep = KeptOutputs::new();
        for e in outputs.iter().filter(|e| e.kind == 0) {
            keep.insert((e.name.clone(), e.index), 0xF);
        }
        plan_merge(&mut keep, &outputs, true, false, false, false);
        let mut wide = outputs.clone();
        merge_rows(&mut wide, &keep);
        assert_eq!(shape(&wide)[2], ("VROW".to_string(), 6, Some(6), 0, 4));
    }

    /// Reordering gives the widest rows the lowest registers, the same in
    /// both stages, and leaves system values where they were.
    #[test]
    fn reordered_rows_put_whole_rows_first() {
        let outputs = vec![
            element("SV_Position", 0, 3, 9, 0, 0, 4),
            element("TEXCOORD", 0, 0, 9, 1, 0, 3),
            element("TEXCOORD", 1, 0, 9, 2, 0, 4),
            element("TEXCOORD", 2, 0, 9, 3, 0, 1),
            element("TEXCOORD", 3, 0, 9, 4, 0, 3),
            element("TEXCOORD", 4, 0, 9, 4, 3, 1),
        ];
        let mut keep = KeptOutputs::new();
        for e in outputs.iter().filter(|e| e.kind == 0) {
            keep.insert((e.name.clone(), e.index), 0xF);
        }
        plan_merge(&mut keep, &outputs, false, true, false, false);
        let finish = |mut els: Vec<SigElement>| -> Vec<(String, u32, Option<u32>, u8)> {
            merge_rows(&mut els, &keep);
            renumber_rows(&mut els, &keep);
            els.iter().map(|e| (e.name.clone(), e.index, e.row, e.cols)).collect()
        };
        let mesh = finish(outputs.clone());
        assert_eq!(
            mesh,
            vec![
                ("SV_Position".to_string(), 0, Some(0), 4),
                ("TEXCOORD".to_string(), 1, Some(1), 4),
                ("VROW".to_string(), 4, Some(2), 4),
                ("TEXCOORD".to_string(), 0, Some(3), 3),
                ("TEXCOORD".to_string(), 2, Some(4), 1),
            ]
        );
        let pixel = finish(vec![outputs[5].clone(), outputs[3].clone(), outputs[4].clone(), outputs[1].clone(), outputs[2].clone()]);
        assert_eq!(pixel, mesh[1..]);
    }

    /// With distances last, a row of clip distances takes the last register
    /// in both stages, the arbitrary rows the ones before it (widest first),
    /// and the signature lists them in that order; without, it stays.
    #[test]
    fn clip_distances_take_the_last_register() {
        let outputs = vec![
            element("SV_Position", 0, 3, 9, 0, 0, 4),
            element("SV_ClipDistance", 0, 6, 9, 1, 0, 4),
            element("TEXCOORD", 0, 0, 9, 2, 0, 4),
            element("TEXCOORD", 1, 0, 9, 3, 0, 3),
            element("TEXCOORD", 2, 0, 9, 4, 0, 4),
            element("TEXCOORD", 3, 0, 9, 5, 0, 2),
        ];
        let reads = || {
            let mut keep = KeptOutputs::new();
            for e in outputs.iter().filter(|e| e.kind == 0) {
                keep.insert((e.name.to_ascii_uppercase(), e.index), 0xF);
            }
            keep
        };
        let shape = |keep: &KeptOutputs, mut els: Vec<SigElement>| -> Vec<(String, Option<u32>)> {
            merge_rows(&mut els, keep);
            renumber_rows(&mut els, keep);
            els.iter().map(|e| (e.name.clone(), e.row)).collect()
        };
        let mut keep = reads();
        plan_merge(&mut keep, &outputs, false, true, true, false);
        let mesh = shape(&keep, outputs.clone());
        let row = |name: &str, r: u32| (name.to_string(), Some(r));
        assert_eq!(
            mesh,
            vec![
                row("SV_Position", 0),
                row("TEXCOORD", 1),
                row("TEXCOORD", 2),
                row("TEXCOORD", 3),
                row("TEXCOORD", 4),
                row("SV_ClipDistance", 5),
            ]
        );
        let registers = |els: &[SigElement]| -> Vec<(u32, Option<u32>)> {
            let mut els = els.to_vec();
            merge_rows(&mut els, &keep);
            renumber_rows(&mut els, &keep);
            els.iter().map(|e| (e.index, e.row)).collect()
        };
        // The float4 rows first, then xyz, then xy.
        assert_eq!(
            registers(&outputs[2..]),
            vec![(0, Some(1)), (2, Some(2)), (1, Some(3)), (3, Some(4))]
        );
        // The pixel shader declares the clip distance and its inputs in
        // another order and without the position: the same registers.
        let pixel = vec![outputs[4].clone(), outputs[1].clone(), outputs[2].clone(), outputs[5].clone(), outputs[3].clone()];
        assert_eq!(shape(&keep, pixel), mesh[1..]);
        // Level 3 leaves the clip distance at its register.
        let mut keep = reads();
        plan_merge(&mut keep, &outputs, false, true, false, false);
        assert_eq!(shape(&keep, outputs.clone())[1], row("SV_ClipDistance", 1));
    }

    /// Level 5: with clip distances the rows of three components come first,
    /// then two, the float4 rows last and the distances after them, the same
    /// in both stages; without distances the order is level 4's.
    #[test]
    fn with_clip_distances_rows_of_three_come_first() {
        let outputs = vec![
            element("SV_Position", 0, 3, 9, 0, 0, 4),
            element("SV_ClipDistance", 0, 6, 9, 1, 0, 4),
            element("TEXCOORD", 0, 0, 9, 2, 0, 4),
            element("TEXCOORD", 1, 0, 9, 3, 0, 3),
            element("TEXCOORD", 2, 0, 9, 4, 0, 4),
            element("TEXCOORD", 3, 0, 9, 5, 0, 2),
        ];
        let plan = |outputs: &[SigElement], xyz_first: bool| {
            let mut keep = KeptOutputs::new();
            for e in outputs.iter().filter(|e| e.kind == 0) {
                keep.insert((e.name.to_ascii_uppercase(), e.index), 0xF);
            }
            plan_merge(&mut keep, outputs, false, true, true, xyz_first);
            keep
        };
        let shape = |keep: &KeptOutputs, mut els: Vec<SigElement>| -> Vec<(String, u32, Option<u32>)> {
            merge_rows(&mut els, keep);
            renumber_rows(&mut els, keep);
            els.iter().map(|e| (e.name.clone(), e.index, e.row)).collect()
        };
        let keep = plan(&outputs, true);
        let mesh = shape(&keep, outputs.clone());
        let el = |name: &str, index: u32, r: u32| (name.to_string(), index, Some(r));
        assert_eq!(
            mesh,
            vec![
                el("SV_Position", 0, 0),
                el("TEXCOORD", 1, 1),
                el("TEXCOORD", 3, 2),
                el("TEXCOORD", 0, 3),
                el("TEXCOORD", 2, 4),
                el("SV_ClipDistance", 0, 5),
            ]
        );
        let pixel = vec![outputs[4].clone(), outputs[1].clone(), outputs[2].clone(), outputs[5].clone(), outputs[3].clone()];
        assert_eq!(shape(&keep, pixel), mesh[1..]);
        // No distances: as level 4, widest first.
        let plain: Vec<SigElement> = outputs.iter().filter(|e| e.kind != 6).cloned().collect();
        let keep = plan(&plain, true);
        assert_eq!(shape(&keep, plain.clone()), shape(&plan(&plain, false), plain.clone()));
        assert_eq!(shape(&keep, plain.clone())[1], el("TEXCOORD", 0, 2));
    }
}
