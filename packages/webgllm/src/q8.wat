(module
  ;; Memory: [0, 256 KiB) every half's value as f32 (the caller's halfTable), then the caller's buffers.
  (memory (export "mem") 5)

  ;; toHalf (half.ts): f32 to half bits, rounding to nearest even.
  (func $toHalf (param $x f32) (result i32)
    (local $bits i32) (local $sign i32) (local $exp i32) (local $mant i32) (local $e i32)
    (local $shift i32) (local $half i32) (local $rest i32) (local $h i32)
    (local.set $bits (i32.reinterpret_f32 (local.get $x)))
    (local.set $sign (i32.and (i32.shr_u (local.get $bits) (i32.const 16)) (i32.const 0x8000)))
    (local.set $exp (i32.and (i32.shr_u (local.get $bits) (i32.const 23)) (i32.const 0xff)))
    (local.set $mant (i32.and (local.get $bits) (i32.const 0x7fffff)))
    (if (i32.eq (local.get $exp) (i32.const 0xff))
      (then (return (i32.or (i32.or (local.get $sign) (i32.const 0x7c00))
        (select (i32.const 0x200) (i32.const 0) (local.get $mant))))))
    (local.set $e (i32.sub (local.get $exp) (i32.const 112)))
    (if (i32.ge_s (local.get $e) (i32.const 31))
      (then (return (i32.or (local.get $sign) (i32.const 0x7c00)))))
    (if (i32.le_s (local.get $e) (i32.const 0))
      (then
        (if (i32.lt_s (local.get $e) (i32.const -10)) (then (return (local.get $sign))))
        (local.set $mant (i32.or (local.get $mant) (i32.const 0x800000)))
        (local.set $shift (i32.sub (i32.const 14) (local.get $e)))
        (local.set $half (i32.shl (i32.const 1) (i32.sub (local.get $shift) (i32.const 1))))
        (local.set $rest (i32.and (local.get $mant) (i32.sub (i32.shl (i32.const 1) (local.get $shift)) (i32.const 1))))
        (local.set $h (i32.shr_u (local.get $mant) (local.get $shift)))
        (if (i32.or (i32.gt_u (local.get $rest) (local.get $half))
                    (i32.and (i32.eq (local.get $rest) (local.get $half)) (i32.and (local.get $h) (i32.const 1))))
          (then (local.set $h (i32.add (local.get $h) (i32.const 1)))))
        (return (i32.or (local.get $sign) (local.get $h)))))
    (local.set $h (i32.or (i32.shl (local.get $e) (i32.const 10)) (i32.shr_u (local.get $mant) (i32.const 13))))
    (local.set $rest (i32.and (local.get $mant) (i32.const 0x1fff)))
    (if (i32.or (i32.gt_u (local.get $rest) (i32.const 0x1000))
                (i32.and (i32.eq (local.get $rest) (i32.const 0x1000)) (i32.and (local.get $h) (i32.const 1))))
      (then (local.set $h (i32.add (local.get $h) (i32.const 1)))))
    (i32.or (local.get $sign) (local.get $h)))

  ;; Four values to four codes: ((w / scale + 128.5) | 0) - 128, clamped to -127..127 (quantizeQ8's), as i32 lanes.
  (func $codes (param $w v128) (param $s v128) (result v128)
    (local $lo v128) (local $hi v128)
    (local.set $lo (i32x4.trunc_sat_f64x2_s_zero
      (f64x2.add (f64x2.div (f64x2.promote_low_f32x4 (local.get $w)) (local.get $s)) (f64x2.splat (f64.const 128.5)))))
    (local.set $hi (i32x4.trunc_sat_f64x2_s_zero
      (f64x2.add (f64x2.div (f64x2.promote_low_f32x4
        (i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 (local.get $w) (local.get $w))) (local.get $s))
        (f64x2.splat (f64.const 128.5)))))
    (i32x4.max_s (i32x4.min_s
      (i32x4.sub (i8x16.shuffle 0 1 2 3 4 5 6 7 16 17 18 19 20 21 22 23 (local.get $lo) (local.get $hi)) (i32x4.splat (i32.const 128)))
      (i32x4.splat (i32.const 127))) (i32x4.splat (i32.const -127))))

  ;; One block's 32 values (w0 .. w7, 4 each) as quantizeQ8 writes them: its half scale at $sp, 32 codes at $qp.
  (func $emit (param $w0 v128) (param $w1 v128) (param $w2 v128) (param $w3 v128)
              (param $w4 v128) (param $w5 v128) (param $w6 v128) (param $w7 v128) (param $sp i32) (param $qp i32)
    (local $m v128) (local $max f32) (local $h i32) (local $scale f32) (local $s v128)
    (local.set $m (f32x4.max
      (f32x4.max (f32x4.max (f32x4.abs (local.get $w0)) (f32x4.abs (local.get $w1))) (f32x4.max (f32x4.abs (local.get $w2)) (f32x4.abs (local.get $w3))))
      (f32x4.max (f32x4.max (f32x4.abs (local.get $w4)) (f32x4.abs (local.get $w5))) (f32x4.max (f32x4.abs (local.get $w6)) (f32x4.abs (local.get $w7))))))
    (local.set $max (f32.max (f32.max (f32x4.extract_lane 0 (local.get $m)) (f32x4.extract_lane 1 (local.get $m)))
                             (f32.max (f32x4.extract_lane 2 (local.get $m)) (f32x4.extract_lane 3 (local.get $m)))))
    ;; toHalf(max / 127): the quotient in f64, then f32 (half.ts's f32[0] = value), then half.
    (local.set $h (call $toHalf (f32.demote_f64 (f64.div (f64.promote_f32 (local.get $max)) (f64.const 127)))))
    (i32.store16 (local.get $sp) (local.get $h))
    (local.set $scale (f32.load (i32.shl (local.get $h) (i32.const 2))))
    (if (i32.eqz (f32.gt (local.get $scale) (f32.const 0)))
      (then
        (v128.store (local.get $qp) (v128.const i32x4 0 0 0 0))
        (v128.store offset=16 (local.get $qp) (v128.const i32x4 0 0 0 0))
        (return)))
    (local.set $s (f64x2.splat (f64.promote_f32 (local.get $scale))))
    (v128.store (local.get $qp) (i8x16.narrow_i16x8_s
      (i16x8.narrow_i32x4_s (call $codes (local.get $w0) (local.get $s)) (call $codes (local.get $w1) (local.get $s)))
      (i16x8.narrow_i32x4_s (call $codes (local.get $w2) (local.get $s)) (call $codes (local.get $w3) (local.get $s)))))
    (v128.store offset=16 (local.get $qp) (i8x16.narrow_i16x8_s
      (i16x8.narrow_i32x4_s (call $codes (local.get $w4) (local.get $s)) (call $codes (local.get $w5) (local.get $s)))
      (i16x8.narrow_i32x4_s (call $codes (local.get $w6) (local.get $s)) (call $codes (local.get $w7) (local.get $s))))))

  ;; 16 codes (u8) as floats a · code + c, in four vectors: the first, the rest via $q1 .. $q3 (globals spare locals).
  (func $f0 (param $c v128) (result v128) (f32x4.convert_i32x4_u (i32x4.extend_low_i16x8_u (i16x8.extend_low_i8x16_u (local.get $c)))))
  (func $f1 (param $c v128) (result v128) (f32x4.convert_i32x4_u (i32x4.extend_high_i16x8_u (i16x8.extend_low_i8x16_u (local.get $c)))))
  (func $f2 (param $c v128) (result v128) (f32x4.convert_i32x4_u (i32x4.extend_low_i16x8_u (i16x8.extend_high_i8x16_u (local.get $c)))))
  (func $f3 (param $c v128) (result v128) (f32x4.convert_i32x4_u (i32x4.extend_high_i16x8_u (i16x8.extend_high_i8x16_u (local.get $c)))))

  ;; A Q5_K sub-block (32 codes: $c0 weights 0-15, $c1 16-31): d · scale · code - dmin · min (dequantizeQ5_K's), emitted.
  (func $q5block (param $c0 v128) (param $c1 v128) (param $a f32) (param $b f32) (param $sp i32) (param $qp i32)
    (local $va v128) (local $vb v128)
    (local.set $va (f32x4.splat (local.get $a)))
    (local.set $vb (f32x4.splat (local.get $b)))
    (call $emit
      (f32x4.sub (f32x4.mul (local.get $va) (call $f0 (local.get $c0))) (local.get $vb))
      (f32x4.sub (f32x4.mul (local.get $va) (call $f1 (local.get $c0))) (local.get $vb))
      (f32x4.sub (f32x4.mul (local.get $va) (call $f2 (local.get $c0))) (local.get $vb))
      (f32x4.sub (f32x4.mul (local.get $va) (call $f3 (local.get $c0))) (local.get $vb))
      (f32x4.sub (f32x4.mul (local.get $va) (call $f0 (local.get $c1))) (local.get $vb))
      (f32x4.sub (f32x4.mul (local.get $va) (call $f1 (local.get $c1))) (local.get $vb))
      (f32x4.sub (f32x4.mul (local.get $va) (call $f2 (local.get $c1))) (local.get $vb))
      (f32x4.sub (f32x4.mul (local.get $va) (call $f3 (local.get $c1))) (local.get $vb))
      (local.get $sp) (local.get $qp)))

  ;; get_scale_min_k4: sub-block j's 6-bit scale (low byte) and min (next byte).
  (func $scaleMin (param $sc i32) (param $j i32) (result i32)
    (if (result i32) (i32.lt_u (local.get $j) (i32.const 4))
      (then (i32.or
        (i32.and (i32.load8_u (i32.add (local.get $sc) (local.get $j))) (i32.const 63))
        (i32.shl (i32.and (i32.load8_u (i32.add (local.get $sc) (i32.add (local.get $j) (i32.const 4)))) (i32.const 63)) (i32.const 8))))
      (else (i32.or
        (i32.or (i32.and (i32.load8_u (i32.add (local.get $sc) (i32.add (local.get $j) (i32.const 4)))) (i32.const 15))
                (i32.shl (i32.shr_u (i32.load8_u (i32.add (local.get $sc) (i32.sub (local.get $j) (i32.const 4)))) (i32.const 6)) (i32.const 4)))
        (i32.shl (i32.or (i32.shr_u (i32.load8_u (i32.add (local.get $sc) (i32.add (local.get $j) (i32.const 4)))) (i32.const 4))
                         (i32.shl (i32.shr_u (i32.load8_u (i32.add (local.get $sc) (local.get $j))) (i32.const 6)) (i32.const 4)))
                 (i32.const 8))))))

  ;; Q5_K rows (176 bytes a super-block of 256) at $src into quantizeQ8's layout for rows from a multiple of 4:
  ;; codes at $qout, half scales at $sout (row group by row group: block, then row).
  (func (export "q5k") (param $src i32) (param $qout i32) (param $sout i32) (param $rows i32) (param $sbs i32)
    (local $i i32) (local $sb i32) (local $jj i32) (local $p i32) (local $si i32) (local $blocks i32)
    (local $d f32) (local $dmin f32) (local $ql i32) (local $sm i32) (local $j i32)
    (local $l0 v128) (local $l1 v128) (local $h0 v128) (local $h1 v128) (local $sh i32)
    (local.set $blocks (i32.mul (local.get $sbs) (i32.const 8)))
    (block $rowsDone (loop $row
      (br_if $rowsDone (i32.ge_u (local.get $i) (local.get $rows)))
      (local.set $si (i32.add (i32.mul (i32.mul (i32.shr_u (local.get $i) (i32.const 2)) (local.get $blocks)) (i32.const 4))
                              (i32.and (local.get $i) (i32.const 3))))
      (local.set $sb (i32.const 0))
      (block $sbDone (loop $super
        (br_if $sbDone (i32.ge_u (local.get $sb) (local.get $sbs)))
        (local.set $p (i32.add (local.get $src) (i32.mul (i32.add (i32.mul (local.get $i) (local.get $sbs)) (local.get $sb)) (i32.const 176))))
        (local.set $d (f32.load (i32.shl (i32.load16_u (local.get $p)) (i32.const 2))))
        (local.set $dmin (f32.load (i32.shl (i32.load16_u offset=2 (local.get $p)) (i32.const 2))))
        (local.set $h0 (v128.load offset=16 (local.get $p)))
        (local.set $h1 (v128.load offset=32 (local.get $p)))
        (local.set $jj (i32.const 0))
        (block $pairsDone (loop $pair
          (br_if $pairsDone (i32.ge_u (local.get $jj) (i32.const 4)))
          (local.set $ql (i32.add (i32.add (local.get $p) (i32.const 48)) (i32.mul (local.get $jj) (i32.const 32))))
          (local.set $l0 (v128.load (local.get $ql)))
          (local.set $l1 (v128.load offset=16 (local.get $ql)))
          ;; Even sub-block j = 2jj: low nibbles, qh bit j.
          (local.set $j (i32.shl (local.get $jj) (i32.const 1)))
          (local.set $sm (call $scaleMin (i32.add (local.get $p) (i32.const 4)) (local.get $j)))
          (call $q5block
            (v128.or (v128.and (local.get $l0) (i8x16.splat (i32.const 15)))
                     (i8x16.shl (v128.and (i8x16.shr_u (local.get $h0) (local.get $j)) (i8x16.splat (i32.const 1))) (i32.const 4)))
            (v128.or (v128.and (local.get $l1) (i8x16.splat (i32.const 15)))
                     (i8x16.shl (v128.and (i8x16.shr_u (local.get $h1) (local.get $j)) (i8x16.splat (i32.const 1))) (i32.const 4)))
            (f32.mul (local.get $d) (f32.convert_i32_u (i32.and (local.get $sm) (i32.const 255))))
            (f32.mul (local.get $dmin) (f32.convert_i32_u (i32.shr_u (local.get $sm) (i32.const 8))))
            (i32.add (local.get $sout) (i32.shl (local.get $si) (i32.const 1)))
            (i32.add (local.get $qout) (i32.shl (local.get $si) (i32.const 5))))
          (local.set $si (i32.add (local.get $si) (i32.const 4)))
          ;; Odd sub-block j + 1: high nibbles, qh bit j + 1.
          (local.set $j (i32.add (local.get $j) (i32.const 1)))
          (local.set $sm (call $scaleMin (i32.add (local.get $p) (i32.const 4)) (local.get $j)))
          (call $q5block
            (v128.or (i8x16.shr_u (local.get $l0) (i32.const 4))
                     (i8x16.shl (v128.and (i8x16.shr_u (local.get $h0) (local.get $j)) (i8x16.splat (i32.const 1))) (i32.const 4)))
            (v128.or (i8x16.shr_u (local.get $l1) (i32.const 4))
                     (i8x16.shl (v128.and (i8x16.shr_u (local.get $h1) (local.get $j)) (i8x16.splat (i32.const 1))) (i32.const 4)))
            (f32.mul (local.get $d) (f32.convert_i32_u (i32.and (local.get $sm) (i32.const 255))))
            (f32.mul (local.get $dmin) (f32.convert_i32_u (i32.shr_u (local.get $sm) (i32.const 8))))
            (i32.add (local.get $sout) (i32.shl (local.get $si) (i32.const 1)))
            (i32.add (local.get $qout) (i32.shl (local.get $si) (i32.const 5))))
          (local.set $si (i32.add (local.get $si) (i32.const 4)))
          (local.set $jj (i32.add (local.get $jj) (i32.const 1)))
          (br $pair)))
        (local.set $sb (i32.add (local.get $sb) (i32.const 1)))
        (br $super)))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $row)))
  )

  ;; Q4_1 rows (20 bytes a block of 32: half d, half m, 16 bytes of nibbles; weight = code · d + m) the same way.
  (func (export "q41") (param $src i32) (param $qout i32) (param $sout i32) (param $rows i32) (param $blocks i32)
    (local $i i32) (local $b i32) (local $p i32) (local $si i32) (local $v v128) (local $lo v128) (local $hi v128)
    (local $vd v128) (local $vm v128)
    (block $rowsDone (loop $row
      (br_if $rowsDone (i32.ge_u (local.get $i) (local.get $rows)))
      (local.set $si (i32.add (i32.mul (i32.mul (i32.shr_u (local.get $i) (i32.const 2)) (local.get $blocks)) (i32.const 4))
                              (i32.and (local.get $i) (i32.const 3))))
      (local.set $b (i32.const 0))
      (block $blocksDone (loop $block
        (br_if $blocksDone (i32.ge_u (local.get $b) (local.get $blocks)))
        (local.set $p (i32.add (local.get $src) (i32.mul (i32.add (i32.mul (local.get $i) (local.get $blocks)) (local.get $b)) (i32.const 20))))
        (local.set $vd (f32x4.splat (f32.load (i32.shl (i32.load16_u (local.get $p)) (i32.const 2)))))
        (local.set $vm (f32x4.splat (f32.load (i32.shl (i32.load16_u offset=2 (local.get $p)) (i32.const 2)))))
        (local.set $v (v128.load offset=4 (local.get $p)))
        (local.set $lo (v128.and (local.get $v) (i8x16.splat (i32.const 15))))
        (local.set $hi (i8x16.shr_u (local.get $v) (i32.const 4)))
        (call $emit
          (f32x4.add (f32x4.mul (call $f0 (local.get $lo)) (local.get $vd)) (local.get $vm))
          (f32x4.add (f32x4.mul (call $f1 (local.get $lo)) (local.get $vd)) (local.get $vm))
          (f32x4.add (f32x4.mul (call $f2 (local.get $lo)) (local.get $vd)) (local.get $vm))
          (f32x4.add (f32x4.mul (call $f3 (local.get $lo)) (local.get $vd)) (local.get $vm))
          (f32x4.add (f32x4.mul (call $f0 (local.get $hi)) (local.get $vd)) (local.get $vm))
          (f32x4.add (f32x4.mul (call $f1 (local.get $hi)) (local.get $vd)) (local.get $vm))
          (f32x4.add (f32x4.mul (call $f2 (local.get $hi)) (local.get $vd)) (local.get $vm))
          (f32x4.add (f32x4.mul (call $f3 (local.get $hi)) (local.get $vd)) (local.get $vm))
          (i32.add (local.get $sout) (i32.shl (local.get $si) (i32.const 1)))
          (i32.add (local.get $qout) (i32.shl (local.get $si) (i32.const 5))))
        (local.set $si (i32.add (local.get $si) (i32.const 4)))
        (local.set $b (i32.add (local.get $b) (i32.const 1)))
        (br $block)))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $row)))
  )
)
