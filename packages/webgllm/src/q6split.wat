(module
  ;; repack.ts's q6SplitRows over this memory: row groups of a Q6_K embedding as packQ6_K stores them
  ;; (a group's super-blocks, each row's 208 bytes: ql[128], qh[64], scales[16]) at $src, into the split
  ;; layout: $low RGBA4 texels (group, 32-weight block, row, 8), $high RGBA4 texels (group, block, row, 4),
  ;; $sub RGBA32UI texels (group, super-block, row: its 16 scale bytes).
  (memory (export "mem") 1)

  ;; Bytes 2i and 2i + 1 of a 32-byte run (nibble values) as byte i's low and high nibbles: 16 bytes.
  (func $pairs (param $a v128) (param $b v128) (result v128)
    (v128.or
      (i8x16.shuffle 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 (local.get $a) (local.get $b))
      (i8x16.shl (i8x16.shuffle 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 (local.get $a) (local.get $b)) (i32.const 4))))

  (func (export "split") (param $src i32) (param $low i32) (param $high i32) (param $sub i32) (param $groups i32) (param $sbs i32)
    (local $gl i32) (local $s i32) (local $r i32) (local $b i32) (local $w i32) (local $k i32) (local $row i32)
    (local $ql i32) (local $qh i32) (local $sh i32) (local $l0 v128) (local $l1 v128) (local $m v128) (local $h v128)
    (block $gDone (loop $g
      (br_if $gDone (i32.ge_u (local.get $gl) (local.get $groups)))
      (local.set $s (i32.const 0))
      (block $sDone (loop $super
        (br_if $sDone (i32.ge_u (local.get $s) (local.get $sbs)))
        (local.set $r (i32.const 0))
        (block $rDone (loop $rows
          (br_if $rDone (i32.ge_u (local.get $r) (i32.const 4)))
          ;; This row's 208 bytes, and its index among the group's (super-block, row) pairs.
          (local.set $row (i32.add (i32.mul (i32.add (i32.mul (local.get $gl) (local.get $sbs)) (local.get $s)) (i32.const 4)) (local.get $r)))
          (local.set $w (i32.add (local.get $src) (i32.mul (local.get $row) (i32.const 208))))
          (v128.store (i32.add (local.get $sub) (i32.shl (local.get $row) (i32.const 4))) (v128.load offset=192 (local.get $w)))
          (local.set $b (i32.const 0))
          (block $bDone (loop $blocks
            (br_if $bDone (i32.ge_u (local.get $b) (i32.const 8)))
            ;; Block b: half n = b >> 2, quarter qq = b & 3; global block k = s · 8 + b.
            (local.set $k (i32.add (i32.mul (local.get $s) (i32.const 8)) (local.get $b)))
            (local.set $ql (i32.add (i32.add (local.get $w) (i32.mul (i32.shr_u (local.get $b) (i32.const 2)) (i32.const 64)))
                                    (i32.mul (i32.and (local.get $b) (i32.const 1)) (i32.const 32))))
            (local.set $l0 (v128.load (local.get $ql)))
            (local.set $l1 (v128.load offset=16 (local.get $ql)))
            (if (i32.lt_u (i32.and (local.get $b) (i32.const 3)) (i32.const 2))
              (then
                (local.set $l0 (v128.and (local.get $l0) (i8x16.splat (i32.const 15))))
                (local.set $l1 (v128.and (local.get $l1) (i8x16.splat (i32.const 15)))))
              (else
                (local.set $l0 (i8x16.shr_u (local.get $l0) (i32.const 4)))
                (local.set $l1 (i8x16.shr_u (local.get $l1) (i32.const 4)))))
            ;; 8 low texels: (group, block, row) · 8 u16s.
            (v128.store
              (i32.add (local.get $low) (i32.shl (i32.add (i32.mul (i32.add (i32.mul (i32.mul (local.get $gl) (local.get $sbs)) (i32.const 8)) (local.get $k)) (i32.const 4)) (local.get $r)) (i32.const 4)))
              (call $pairs (local.get $l0) (local.get $l1)))
            ;; 4 high texels: weight j's top 2 bits and j + 16's above them, a nibble each.
            (local.set $qh (i32.add (i32.add (local.get $w) (i32.const 128)) (i32.mul (i32.shr_u (local.get $b) (i32.const 2)) (i32.const 32))))
            (local.set $sh (i32.shl (i32.and (local.get $b) (i32.const 3)) (i32.const 1)))
            (local.set $m (i8x16.splat (i32.const 3)))
            (local.set $h (v128.or
              (v128.and (i8x16.shr_u (v128.load (local.get $qh)) (local.get $sh)) (local.get $m))
              (i8x16.shl (v128.and (i8x16.shr_u (v128.load offset=16 (local.get $qh)) (local.get $sh)) (local.get $m)) (i32.const 2))))
            (v128.store64_lane 0
              (i32.add (local.get $high) (i32.shl (i32.add (i32.mul (i32.add (i32.mul (i32.mul (local.get $gl) (local.get $sbs)) (i32.const 8)) (local.get $k)) (i32.const 4)) (local.get $r)) (i32.const 3)))
              (call $pairs (local.get $h) (local.get $h)))
            (local.set $b (i32.add (local.get $b) (i32.const 1)))
            (br $blocks)))
          (local.set $r (i32.add (local.get $r) (i32.const 1)))
          (br $rows)))
        (local.set $s (i32.add (local.get $s) (i32.const 1)))
        (br $super)))
      (local.set $gl (i32.add (local.get $gl) (i32.const 1)))
      (br $g)))
  )
)
