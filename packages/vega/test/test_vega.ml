(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module S = Vega.Schedule

(* Helpers *)

let f32 = Nx.float32
let vec xs = Nx.create f32 [| Array.length xs |] xs
let mat r c xs = Nx.create f32 [| r; c |] xs
let to_arr t = Nx.to_array (Nx.reshape [| -1 |] t)
let eps = float 1e-5
let lr01 = S.constant 0.1

let converges ~msg ~tol tx =
  let param = ref (vec [| 5.0; -3.0 |]) in
  let st = ref (Vega.init tx !param) in
  for _ = 1 to 200 do
    let p, s = Vega.step !st ~grad:!param ~param:!param in
    param := p;
    st := s
  done;
  let v = to_arr !param in
  equal ~msg:(msg ^ "[0]") (float tol) 0.0 v.(0);
  equal ~msg:(msg ^ "[1]") (float tol) 0.0 v.(1)

(* Schedules *)

let test_polynomial_decay () =
  let s =
    S.polynomial_decay ~init_value:1.0 ~end_value:0.0 ~decay_steps:100 ()
  in
  equal ~msg:"step 0" (float 1e-6) 1.0 (S.eval s 0);
  equal ~msg:"step 50 (power=1, linear)" (float 1e-6) 0.5 (S.eval s 50);
  equal ~msg:"step 100" (float 1e-6) 0.0 (S.eval s 100);
  equal ~msg:"clamps past end" (float 1e-6) 0.0 (S.eval s 200);
  let s2 =
    S.polynomial_decay ~init_value:1.0 ~end_value:0.0 ~decay_steps:100
      ~power:2.0 ()
  in
  equal ~msg:"power=2 at midpoint" (float 1e-6) 0.25 (S.eval s2 50)

let test_warmup_cosine_decay () =
  let s =
    S.warmup_cosine_decay ~init_value:0.0 ~peak_value:1.0 ~warmup_steps:10
      ~decay_steps:90 ()
  in
  equal ~msg:"step 0" (float 1e-6) 0.0 (S.eval s 0);
  equal ~msg:"step 5 (warmup midpoint)" (float 1e-6) 0.5 (S.eval s 5);
  equal ~msg:"step 10 (peak)" (float 1e-6) 1.0 (S.eval s 10);
  equal ~msg:"step 100 (fully decayed)" (float 1e-6) 0.0 (S.eval s 100);
  equal ~msg:"past end" (float 1e-6) 0.0 (S.eval s 200)

let test_piecewise_constant () =
  let s =
    S.piecewise_constant ~boundaries:[ 10; 20 ] ~values:[ 1.0; 0.1; 0.01 ]
  in
  equal ~msg:"segment 1" (float 1e-6) 1.0 (S.eval s 5);
  equal ~msg:"boundary" (float 1e-6) 1.0 (S.eval s 10);
  equal ~msg:"segment 2" (float 1e-6) 0.1 (S.eval s 15);
  equal ~msg:"segment 3" (float 1e-6) 0.01 (S.eval s 25)

let test_piecewise_constant_validation () =
  raises_match Exn.invalid_arg (fun () ->
      ignore (S.piecewise_constant ~boundaries:[ 10 ] ~values:[ 1.0 ] : S.t));
  raises_match Exn.invalid_arg (fun () ->
      ignore
        (S.piecewise_constant ~boundaries:[ 20; 10 ] ~values:[ 1.; 2.; 3. ]
          : S.t))

let test_join () =
  let s =
    S.join [ (10, S.constant 1.0); (10, S.constant 2.0); (10, S.constant 3.0) ]
  in
  equal ~msg:"segment 1" (float 1e-6) 1.0 (S.eval s 5);
  equal ~msg:"segment 2" (float 1e-6) 2.0 (S.eval s 15);
  equal ~msg:"segment 3" (float 1e-6) 3.0 (S.eval s 25);
  equal ~msg:"past end extends last" (float 1e-6) 3.0 (S.eval s 100)

let test_join_step_reset () =
  let calls = ref [] in
  let spy name =
    S.join
      [
        ( 5,
          fun step ->
            calls := (name, Int32.to_int (Nx.item [] step)) :: !calls;
            Nx.scalar Nx.float32 0. );
      ]
  in
  let s = spy "a" in
  ignore (S.eval s 3);
  equal ~msg:"step passed to inner schedule"
    (list (pair string int))
    [ ("a", 3) ]
    (List.rev !calls)

let test_join_validation () =
  raises_match Exn.invalid_arg (fun () -> ignore (S.join [] : S.t));
  raises_match Exn.invalid_arg (fun () ->
      ignore (S.join [ (0, S.constant 1.0) ] : S.t))

let test_cosine_decay_restarts () =
  let s = S.cosine_decay_restarts ~init_value:1.0 ~decay_steps:100 () in
  equal ~msg:"step 0 (peak)" (float 1e-6) 1.0 (S.eval s 0);
  equal ~msg:"step 100 (restart)" (float 1e-6) 1.0 (S.eval s 100);
  equal ~msg:"step 200 (second restart)" (float 1e-6) 1.0 (S.eval s 200);
  equal ~msg:"step 50 (midpoint)" (float 1e-6) 0.5 (S.eval s 50)

let test_cosine_decay_restarts_t_mul () =
  let s =
    S.cosine_decay_restarts ~init_value:1.0 ~decay_steps:10 ~t_mul:2.0 ()
  in
  (* First cycle: 10 steps. Second: 20 steps. *)
  equal ~msg:"step 0 (start)" (float 1e-6) 1.0 (S.eval s 0);
  equal ~msg:"step 10 (second cycle start)" (float 1e-6) 1.0 (S.eval s 10);
  equal ~msg:"step 30 (third cycle start)" (float 1e-6) 1.0 (S.eval s 30)

let test_cosine_decay_restarts_m_mul () =
  let s =
    S.cosine_decay_restarts ~init_value:1.0 ~decay_steps:100 ~m_mul:0.5 ()
  in
  equal ~msg:"cycle 0 peak" (float 1e-6) 1.0 (S.eval s 0);
  equal ~msg:"cycle 1 peak" (float 1e-6) 0.5 (S.eval s 100);
  equal ~msg:"cycle 2 peak" (float 1e-6) 0.25 (S.eval s 200)

let test_one_cycle () =
  let s = S.one_cycle ~max_value:1.0 ~total_steps:100 () in
  (* warmup: 30 steps (pct_start=0.3), init=1/25=0.04, peak=1.0 *)
  equal ~msg:"step 0" (float 1e-6) 0.04 (S.eval s 0);
  equal ~msg:"step 30 (peak)" (float 1e-6) 1.0 (S.eval s 30);
  (* decay: 70 steps, from 1.0 to 1/10000=0.0001 *)
  let end_val = 1.0 /. 10000.0 in
  equal ~msg:"step 100 (end)" (float 1e-6) end_val (S.eval s 100)

(* Schedule property tests — these are `test` values, placed directly in the
   group list below. *)

(* Primitives *)

let test_scale () =
  let tx = Vega.scale 2.0 in
  let grad = vec [| 1.0; -0.5 |] in
  let param = vec [| 0.; 0. |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  equal ~msg:"scaled" (array eps) [| 2.0; -1.0 |] (to_arr upd)

let test_scale_by_schedule () =
  let tx = Vega.scale_by_schedule (S.constant 3.0) in
  let grad = vec [| 1.0; 2.0 |] in
  let param = vec [| 0.; 0. |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  equal ~msg:"scheduled" (array eps) [| 3.0; 6.0 |] (to_arr upd)

let test_scale_by_learning_rate () =
  let tx = Vega.scale_by_learning_rate (S.constant 0.1) in
  let grad = vec [| 10.0; -5.0 |] in
  let param = vec [| 0.; 0. |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  (* negated: updates = grad * (-0.1) *)
  equal ~msg:"negated lr" (array eps) [| -1.0; 0.5 |] (to_arr upd)

let test_trace () =
  let tx = Vega.trace ~decay:0.9 () in
  let grad = vec [| 1.0; 2.0 |] in
  let param = vec [| 0.; 0. |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  (* step 1: vel = 0.9*0 + grad = grad; output = vel *)
  equal ~msg:"step 1" (array eps) [| 1.0; 2.0 |] (to_arr upd)

let test_trace_nesterov () =
  let tx = Vega.trace ~decay:0.9 ~nesterov:true () in
  let grad = vec [| 1.0; 2.0 |] in
  let param = vec [| 0.; 0. |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  (* vel = grad; nesterov output = grad + 0.9 * vel = grad + 0.9*grad =
     1.9*grad *)
  equal ~msg:"nesterov" (array eps) [| 1.9; 3.8 |] (to_arr upd)

let test_add_decayed_weights () =
  let tx = Vega.add_decayed_weights ~rate:(S.constant 0.1) () in
  let grad = vec [| 1.0; 0.0 |] in
  let param = vec [| 10.0; -5.0 |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  (* updates + 0.1 * param *)
  equal ~msg:"wd" (array eps) [| 2.0; -0.5 |] (to_arr upd)

let test_add_decayed_weights_scheduled () =
  let rate step = Nx.mul_s (Nx.cast Nx.float32 step) 0.01 in
  let tx = Vega.add_decayed_weights ~rate () in
  let grad = vec [| 0.0 |] in
  let param = vec [| 10.0 |] in
  let st = Vega.init tx param in
  let upd1, st = Vega.update st ~grad ~param in
  (* step 1: rate=0.01, updates = 0 + 0.01*10 = 0.1 *)
  equal ~msg:"step 1" (array eps) [| 0.1 |] (to_arr upd1);
  let upd2, _ = Vega.update st ~grad ~param in
  (* step 2: rate=0.02, updates = 0 + 0.02*10 = 0.2 *)
  equal ~msg:"step 2" (array eps) [| 0.2 |] (to_arr upd2)

let test_clip () =
  let tx = Vega.clip 1.0 in
  let grad = vec [| 5.0; -0.5; -3.0 |] in
  let param = vec [| 0.; 0.; 0. |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  equal ~msg:"clipped" (array eps) [| 1.0; -0.5; -1.0 |] (to_arr upd)

let test_clip_by_norm () =
  (* norm of [3, 4] = 5, clip to 2.5 → scale by 0.5 *)
  let tx = Vega.clip_by_norm 2.5 in
  let grad = vec [| 3.0; 4.0 |] in
  let param = vec [| 0.; 0. |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  equal ~msg:"rescaled" (array eps) [| 1.5; 2.0 |] (to_arr upd)

let test_clip_by_norm_no_op () =
  let tx = Vega.clip_by_norm 10.0 in
  let grad = vec [| 1.0; 1.0 |] in
  let param = vec [| 0.; 0. |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  equal ~msg:"unchanged" (array eps) [| 1.0; 1.0 |] (to_arr upd)

let test_trust_ratio () =
  let tx = Vega.scale_by_trust_ratio () in
  let grad = vec [| 1.0; 0.0 |] in
  let param = vec [| 3.0; 4.0 |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  (* ||param|| = 5, ||grad|| = 1, ratio = 5/1 = 5 *)
  equal ~msg:"ratio" (array (float 1e-4)) [| 5.0; 0.0 |] (to_arr upd)

let test_trust_ratio_zero_param () =
  let tx = Vega.scale_by_trust_ratio () in
  let grad = vec [| 1.0 |] in
  let param = vec [| 0.0 |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  (* zero param norm → ratio = 1 *)
  equal ~msg:"fallback" (array eps) [| 1.0 |] (to_arr upd)

(* Gradient processing *)

let test_centralize_2d () =
  let tx = Vega.centralize in
  (* 2x3 matrix: row 0 = [1,2,3] mean=2, row 1 = [4,5,6] mean=5 *)
  let grad = mat 2 3 [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let param = mat 2 3 [| 0.; 0.; 0.; 0.; 0.; 0. |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  equal ~msg:"centralized" (array eps)
    [| -1.; 0.; 1.; -1.; 0.; 1. |]
    (to_arr upd)

let test_centralize_1d () =
  let tx = Vega.centralize in
  let grad = vec [| 1.; 2.; 3. |] in
  let param = vec [| 0.; 0.; 0. |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  equal ~msg:"1d unchanged" (array eps) [| 1.; 2.; 3. |] (to_arr upd)

let test_add_noise () =
  let tx = Vega.add_noise ~eta:(S.constant 1.0) () in
  let grad = vec [| 0.; 0. |] in
  let param = vec [| 0.; 0. |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  let v = to_arr upd in
  (* With zero grad, output is pure noise — should be non-zero with high prob *)
  is_true ~msg:"noise injected"
    (Float.abs v.(0) > 1e-10 || Float.abs v.(1) > 1e-10)

(* Adam variants *)

let test_scale_by_adam_step1 () =
  let tx = Vega.scale_by_adam ~b1:0.9 ~b2:0.999 ~eps:1e-8 () in
  let grad = vec [| 2.0 |] in
  let param = vec [| 0. |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  (* mu = 0.1*2 = 0.2, nu = 0.001*4 = 0.004 bc1 = 0.1, bc2 = 0.001 m_hat =
     0.2/0.1 = 2.0, v_hat = 0.004/0.001 = 4.0 out = 2 / (sqrt(4) + 1e-8) = 2/2 =
     1.0 *)
  equal ~msg:"adam step 1" (array (float 1e-4)) [| 1.0 |] (to_arr upd)

let test_amsgrad () =
  let tx = Vega.scale_by_adam ~amsgrad:true () in
  let param = vec [| 0. |] in
  let st = Vega.init tx param in
  (* Step 1: large gradient → large v *)
  let _, st = Vega.update st ~grad:(vec [| 10.0 |]) ~param in
  (* Step 2: small gradient → v decreases, but v_max holds *)
  let _, st = Vega.update st ~grad:(vec [| 0.01 |]) ~param in
  let _, tensors = Vega.state_to_tensors st in
  let nu = to_arr tensors.(1) in
  let v_max = to_arr tensors.(2) in
  is_true ~msg:"v_max >= nu" (v_max.(0) >= nu.(0))

let test_nesterov_differs () =
  let tx_std = Vega.scale_by_adam () in
  let tx_nes = Vega.scale_by_adam ~nesterov:true () in
  let grad = vec [| 3.0; -1.0 |] in
  let param = vec [| 0.; 0. |] in
  let upd_std, _ = Vega.update (Vega.init tx_std param) ~grad ~param in
  let upd_nes, _ = Vega.update (Vega.init tx_nes param) ~grad ~param in
  let a = to_arr upd_std and b = to_arr upd_nes in
  is_true ~msg:"nesterov differs from standard"
    (Float.abs (a.(0) -. b.(0)) > 1e-6)

(* Muon *)

(* A matrix whose singular values span an order of magnitude, so the
   orthogonalization is non-trivial. Its transpose exercises the other
   association of [X X^T X] in the Newton-Schulz iteration. *)
let muon_matrix =
  mat 3 5
    [|
      1.0;
      2.0;
      3.0;
      4.0;
      0.5;
      1.5;
      -2.0;
      0.25;
      1.0;
      -1.0;
      2.5;
      0.75;
      -0.5;
      1.25;
      3.0;
    |]

(* The reference implementation of Newton-Schulz 5, transcribed from the
   published code: transpose so that there are at most as many rows as columns,
   then iterate with [X X^T] throughout. Vega chooses the narrower of [X X^T]
   and [X^T X] instead, so this reaches the same polynomial by another route. *)
let reference_ns5 ?(steps = 5) (x : Nx.float32_t) =
  let rows = Nx.dim 0 x and cols = Nx.dim 1 x in
  let tall = rows > cols in
  let z = if tall then Nx.transpose x else x in
  let norm = Nx.sqrt (Nx.sum (Nx.square z)) in
  let ns = ref (Nx.div z (Nx.add_s norm 1e-7)) in
  for _ = 1 to steps do
    let xk = !ns in
    let a = Nx.matmul xk (Nx.transpose xk) in
    let b = Nx.add (Nx.mul_s a (-4.7750)) (Nx.mul_s (Nx.matmul a a) 2.0315) in
    ns := Nx.add (Nx.mul_s xk 3.4445) (Nx.matmul b xk)
  done;
  if tall then Nx.transpose !ns else !ns

(* One eager step of Muon's chain, returning the raw update. *)
let muon_update ?(scaling = `Update_rms 1.0) ?(momentum = 0.0)
    ?(nesterov = false) g =
  let tx = Vega.scale_by_muon ~momentum ~nesterov ~scaling () in
  fst (Vega.update (Vega.init tx g) ~grad:g ~param:g)

let test_muon_matches_reference_ns5 () =
  List.iter
    (fun (name, g) ->
      let rows = Nx.dim 0 g and cols = Nx.dim 1 g in
      let expected =
        Nx.mul_s (reference_ns5 g) (sqrt (Stdlib.float (max rows cols)))
      in
      equal ~msg:name
        (array (float 1e-5))
        (to_arr expected)
        (to_arr (muon_update g)))
    [ ("wide", muon_matrix); ("tall", Nx.transpose muon_matrix) ]

let test_muon_transpose_equivariant () =
  (* [X X^T] and [X^T X] generate the same quintic, so orthogonalizing a matrix
     and its transpose gives transposes. *)
  let tall = Nx.transpose (muon_update (Nx.transpose muon_matrix)) in
  equal ~msg:"transpose"
    (array (float 1e-6))
    (to_arr (Nx.contiguous tall))
    (to_arr (muon_update muon_matrix))

let test_muon_orthogonalizes () =
  (* Five Newton-Schulz steps approximate the polar factor rather than reach it:
     the update's singular values land in the [0.5, 1.5] window the reference
     implementation accepts, so its rows are approximately orthonormal. *)
  let u = Nx.mul_s (muon_update muon_matrix) (1.0 /. sqrt 5.0) in
  let gram = Nx.matmul u (Nx.transpose u) in
  let deviation = Nx.sub gram (Nx.eye f32 3) in
  is_true ~msg:"semi-orthogonal up to the iteration's error"
    (Nx.item [] (Nx.max (Nx.abs deviation)) < 0.5)

let test_muon_isotropic_spectrum () =
  (* Equal nonzero singular values are each mapped through the quintic
     polynomial, so the update is the polar direction times [phi^5 (1/sqrt 2)]
     up to the rescaling by shape. *)
  let iso = mat 2 3 [| 1.0; 0.0; 0.0; 0.0; 1.0; 0.0 |] in
  let phi x =
    (3.4445 *. x) -. (4.7750 *. x *. x *. x) +. (2.0315 *. (x ** 5.))
  in
  let s = ref (1.0 /. sqrt 2.0) in
  for _ = 1 to 5 do
    s := phi !s
  done;
  equal ~msg:"phi^5 on the spectrum"
    (array (float 1e-5))
    (to_arr (Nx.mul_s iso (!s *. sqrt 3.0)))
    (to_arr (muon_update iso))

let test_muon_scale_conventions () =
  (* The two shape conventions in the units they promise: [`Update_rms r] gives
     the update an RMS of [r] whatever the shape — the ideal polar factor's
     would be exactly [r], and five steps land within a few percent of it —
     while [`Width] gives it [1 /. sqrt cols], independent of the row count. *)
  let rms t = sqrt (Nx.item [] (Nx.mean (Nx.square t))) in
  List.iter
    (fun (name, g) ->
      let cols = Nx.dim 1 g in
      equal ~msg:(name ^ " update rms") (float 0.02) 0.2
        (rms (muon_update ~scaling:(`Update_rms 0.2) g));
      equal ~msg:(name ^ " width scaling") (float 0.015)
        (1.0 /. sqrt (Stdlib.float cols))
        (rms (muon_update ~scaling:`Width g)))
    [ ("wide", muon_matrix); ("tall", Nx.transpose muon_matrix) ]

let test_muon_momentum () =
  (* The buffer is an EMA of the updates — [0.09 * g1 + 0.1 * g2] after two
     steps — and Nesterov interpolates it with the current update before the
     orthogonalization, which changes the direction once the two differ. *)
  let g2 =
    Nx.transpose (mat 5 3 (Array.init 15 (fun i -> float_of_int i -. 7.0)))
  in
  let run ~nesterov =
    let tx = Vega.scale_by_muon ~momentum:0.9 ~nesterov () in
    let param = mat 3 5 (Array.make 15 0.0) in
    let st = Vega.init tx param in
    let _, st = Vega.update st ~grad:muon_matrix ~param in
    let updates, st = Vega.update st ~grad:g2 ~param in
    let _, tensors = Vega.state_to_tensors st in
    (updates, tensors.(0))
  in
  let nesterov, buffer = run ~nesterov:true in
  let plain, _ = run ~nesterov:false in
  let expected = Nx.add (Nx.mul_s muon_matrix 0.09) (Nx.mul_s g2 0.1) in
  equal ~msg:"momentum buffer"
    (array (float 1e-6))
    (to_arr expected) (to_arr buffer);
  is_true ~msg:"nesterov changes the direction" (to_arr nesterov <> to_arr plain)

let test_muon_flattens_higher_dimensions () =
  (* A leaf of more than two dimensions is orthogonalized as the matrix [(dim 0)
     x (product of the rest)] — the view the reference implementation uses for
     convolutional filters — and comes back in its own shape. *)
  let filter =
    Nx.reshape [| 4; 2; 3; 3 |]
      (mat 4 18
         (Array.init 72 (fun i ->
              let i = Stdlib.float_of_int i in
              (sin (i *. 0.7) *. 2.0) +. (cos (i *. 0.31) *. 0.5))))
  in
  let updates = muon_update filter in
  equal ~msg:"the update keeps the leaf's shape" (array int) [| 4; 2; 3; 3 |]
    (Nx.shape updates);
  equal ~msg:"flattened like the reference implementation"
    (array (float 1e-6))
    (to_arr (muon_update (Nx.reshape [| 4; 18 |] filter)))
    (to_arr (Nx.reshape [| 4; 18 |] updates))

let test_muon_converges () =
  (* Muon descends: with the polar factor of the gradient every singular value
     of the residual shrinks by [lr] a step, so a matrix quadratic converges
     fast (and this checks the sign of the update). *)
  let target =
    mat 4 4
      [|
        0.5;
        -1.0;
        0.25;
        2.0;
        1.0;
        0.0;
        -0.5;
        0.75;
        1.5;
        -0.25;
        0.5;
        1.0;
        -1.25;
        0.5;
        0.25;
        -0.75;
      |]
  in
  let param = ref (mat 4 4 (Array.make 16 0.0)) in
  let tx =
    Vega.muon ~momentum:0.0 ~nesterov:false ~scaling:(`Update_rms 1.0)
      (S.constant 0.1)
  in
  let st = ref (Vega.init tx !param) in
  let residual p = sqrt (Nx.item [] (Nx.sum (Nx.square (Nx.sub p target)))) in
  let start = residual !param in
  for _ = 1 to 100 do
    let grads = Nx.mul_s (Nx.sub !param target) 2.0 in
    let p, s = Vega.step !st ~grad:grads ~param:!param in
    param := p;
    st := s
  done;
  is_true ~msg:"reaches the bottom of the matrix quadratic"
    (residual !param < 0.2 *. start)

(* Optimizer convergence *)

let test_lion_converges () = converges ~msg:"lion" ~tol:1.0 (Vega.lion lr01)
let test_radam_converges () = converges ~msg:"radam" ~tol:0.5 (Vega.radam lr01)

let test_adan_converges () =
  converges ~msg:"adan" ~tol:1.0 (Vega.adan (S.constant 0.05))

let test_lamb_converges () = converges ~msg:"lamb" ~tol:0.5 (Vega.lamb lr01)

let test_lars_converges () =
  converges ~msg:"lars" ~tol:0.5 (Vega.lars (S.constant 0.05))

let test_adafactor_converges () =
  (* Adafactor includes its own LR (eps_scale/sqrt(step) ≈ 0.001/sqrt(t)).
     Cumulative displacement after N steps is ~0.002*sqrt(N), so use small
     initial values and 2D shape to exercise the factored path. *)
  let tx = Vega.adafactor () in
  let param = ref (mat 2 2 [| 0.1; -0.05; 0.08; -0.03 |]) in
  let st = ref (Vega.init tx !param) in
  for _ = 1 to 5000 do
    let p, s = Vega.step !st ~grad:!param ~param:!param in
    param := p;
    st := s
  done;
  let v = to_arr !param in
  Array.iter
    (fun x -> is_true ~msg:"adafactor converges" (Float.abs x < 0.05))
    v

let test_adam_amsgrad_converges () =
  converges ~msg:"adam+amsgrad" ~tol:0.5
    (Vega.adam ~b1:0.9 ~b2:0.999 ~eps:1e-8 lr01)

(* Chain composition *)

let test_chain_associativity () =
  let a = Vega.scale 2.0 in
  let b = Vega.clip 5.0 in
  let c = Vega.scale 0.5 in
  let tx1 = Vega.chain [ Vega.chain [ a; b ]; c ] in
  let tx2 = Vega.chain [ a; b; c ] in
  let grad = vec [| 3.0; -4.0 |] in
  let param = vec [| 0.; 0. |] in
  let upd1, _ = Vega.update (Vega.init tx1 param) ~grad ~param in
  let upd2, _ = Vega.update (Vega.init tx2 param) ~grad ~param in
  equal ~msg:"associative" (array eps) (to_arr upd1) (to_arr upd2)

let test_chain_identity () =
  let tx = Vega.scale_by_adam () in
  let tx_wrapped = Vega.chain [ tx ] in
  let grad = vec [| 1.0; -2.0 |] in
  let param = vec [| 0.; 0. |] in
  let upd1, _ = Vega.update (Vega.init tx param) ~grad ~param in
  let upd2, _ = Vega.update (Vega.init tx_wrapped param) ~grad ~param in
  equal ~msg:"identity" (array eps) (to_arr upd1) (to_arr upd2)

let test_chain_ordering_matters () =
  let tx1 = Vega.chain [ Vega.clip 1.0; Vega.scale 10.0 ] in
  let tx2 = Vega.chain [ Vega.scale 10.0; Vega.clip 1.0 ] in
  let grad = vec [| 0.5 |] in
  let param = vec [| 0. |] in
  let upd1, _ = Vega.update (Vega.init tx1 param) ~grad ~param in
  let upd2, _ = Vega.update (Vega.init tx2 param) ~grad ~param in
  (* clip then scale: 0.5 → 0.5 → 5.0 ; scale then clip: 0.5 → 5.0 → 1.0 *)
  equal ~msg:"clip then scale" (array eps) [| 5.0 |] (to_arr upd1);
  equal ~msg:"scale then clip" (array eps) [| 1.0 |] (to_arr upd2)

(* apply_if_finite *)

let test_finite_passes_through () =
  let inner = Vega.scale 2.0 in
  let tx = Vega.apply_if_finite inner in
  let grad = vec [| 1.0; -0.5 |] in
  let param = vec [| 0.; 0. |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  equal ~msg:"pass-through" (array eps) [| 2.0; -1.0 |] (to_arr upd)

let test_nan_skipped () =
  let inner = Vega.scale 1.0 in
  let tx = Vega.apply_if_finite inner in
  let param = vec [| 0.; 0. |] in
  let grad = vec [| Float.nan; 1.0 |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  let v = to_arr upd in
  equal ~msg:"nan → zero[0]" (float 1e-6) 0.0 v.(0);
  equal ~msg:"nan → zero[1]" (float 1e-6) 0.0 v.(1)

let test_inf_skipped () =
  let inner = Vega.scale 1.0 in
  let tx = Vega.apply_if_finite inner in
  let param = vec [| 0.; 0. |] in
  let grad = vec [| Float.infinity; 1.0 |] in
  let upd, _ = Vega.update (Vega.init tx param) ~grad ~param in
  let v = to_arr upd in
  equal ~msg:"inf → zero" (float 1e-6) 0.0 v.(0)

let test_nonfinite_counter () =
  let inner = Vega.scale 1.0 in
  let tx = Vega.apply_if_finite inner in
  let param = vec [| 0. |] in
  let nan_grad = vec [| Float.nan |] in
  let st = Vega.init tx param in
  let _, st = Vega.update st ~grad:nan_grad ~param in
  let _, st = Vega.update st ~grad:nan_grad ~param in
  let _, tensors = Vega.state_to_tensors st in
  (* Last tensor is the counter *)
  let counter = Nx.item [] tensors.(Array.length tensors - 1) in
  equal ~msg:"2 consecutive non-finite" (float 1e-6) 2.0 counter

(* Serialization *)

let test_n_tensors () =
  equal ~msg:"sgd" int 0 (Vega.n_tensors (Vega.sgd lr01));
  equal ~msg:"sgd+momentum" int 1 (Vega.n_tensors (Vega.sgd ~momentum:0.9 lr01));
  equal ~msg:"adam" int 2 (Vega.n_tensors (Vega.adam lr01));
  equal ~msg:"adam+amsgrad" int 3
    (Vega.n_tensors
       (Vega.chain
          [
            Vega.scale_by_adam ~amsgrad:true ();
            Vega.scale_by_learning_rate lr01;
          ]));
  equal ~msg:"lion" int 1 (Vega.n_tensors (Vega.lion lr01));
  equal ~msg:"adan" int 4 (Vega.n_tensors (Vega.adan lr01));
  equal ~msg:"adafactor" int 2 (Vega.n_tensors (Vega.adafactor ()))

let test_serialization_round_trip () =
  let optimizers =
    [
      ("adam", Vega.adam lr01);
      ("adamw", Vega.adamw lr01);
      ("lion", Vega.lion lr01);
      ("radam", Vega.radam lr01);
    ]
  in
  List.iter
    (fun (name, tx) ->
      let param = vec [| 3.0; -2.0 |] in
      let grad = vec [| 1.0; -1.0 |] in
      (* Step once *)
      let st = Vega.init tx param in
      let _, st = Vega.update st ~grad ~param in
      (* Serialize and deserialize *)
      let count, tensors = Vega.state_to_tensors st in
      let st2 = Vega.state_of_tensors tx ~count tensors in
      (* Step again from both *)
      let upd1, _ = Vega.update st ~grad ~param in
      let upd2, _ = Vega.update st2 ~grad ~param in
      equal ~msg:(name ^ " round-trip") (array eps) (to_arr upd1) (to_arr upd2))
    optimizers

let test_wrong_tensor_count () =
  let tx = Vega.adam lr01 in
  raises_match Exn.invalid_arg (fun () ->
      ignore (Vega.state_of_tensors tx ~count:1 [| vec [| 0. |] |]))

(* Validation *)

let test_validation () =
  raises_match Exn.invalid_arg (fun () ->
      ignore (Vega.scale_by_lion ~b1:1.0 ()));
  raises_match Exn.invalid_arg (fun () ->
      ignore (Vega.scale_by_lion ~b2:(-0.1) ()));
  raises_match Exn.invalid_arg (fun () ->
      ignore (Vega.scale_by_adan ~b1:1.0 ()));
  raises_match Exn.invalid_arg (fun () ->
      ignore (Vega.scale_by_adan ~b2:(-0.1) ()));
  raises_match Exn.invalid_arg (fun () ->
      ignore (Vega.scale_by_adan ~b3:1.0 ()));
  raises_match Exn.invalid_arg (fun () ->
      ignore (Vega.adan ~weight_decay:(-1.) lr01));
  raises_match Exn.invalid_arg (fun () ->
      ignore (Vega.scale_by_muon ~momentum:1.0 ()));
  raises_match Exn.invalid_arg (fun () ->
      ignore (Vega.scale_by_muon ~ns_steps:0 ()));
  raises_match Exn.invalid_arg (fun () ->
      ignore (Vega.scale_by_muon ~scaling:(`Update_rms 0.0) ()));
  raises_match Exn.invalid_arg (fun () ->
      ignore (Vega.init (Vega.scale_by_muon ()) (vec [| 1.0 |])));
  raises_match Exn.invalid_arg (fun () ->
      ignore (S.cosine_decay_restarts ~init_value:1. ~decay_steps:0 () : S.t));
  raises_match Exn.invalid_arg (fun () ->
      ignore (S.one_cycle ~max_value:1. ~total_steps:0 () : S.t))

(* Entry point *)

let () =
  run "Vega"
    [
      group "schedule"
        [
          test "polynomial_decay" test_polynomial_decay;
          test "warmup_cosine_decay" test_warmup_cosine_decay;
          test "piecewise_constant" test_piecewise_constant;
          test "piecewise_constant validation"
            test_piecewise_constant_validation;
          test "join" test_join;
          test "join step reset" test_join_step_reset;
          test "join validation" test_join_validation;
          test "cosine_decay_restarts" test_cosine_decay_restarts;
          test "cosine_decay_restarts t_mul" test_cosine_decay_restarts_t_mul;
          test "cosine_decay_restarts m_mul" test_cosine_decay_restarts_m_mul;
          test "one_cycle" test_one_cycle;
          prop "constant is constant"
            Gen.(pair float nat)
            (fun (v, step) ->
              let s = S.constant v in
              equal float_exact (S.eval s 0) (S.eval s step));
          prop "cosine_decay bounded" Gen.nat (fun step ->
              let s = S.cosine_decay ~init_value:1.0 ~decay_steps:100 () in
              let v = S.eval s step in
              is_true ~msg:">=0" (v >= 0.0);
              is_true ~msg:"<=1" (v <= 1.0 +. 1e-6));
          prop "one_cycle bounded" Gen.nat (fun step ->
              let s = S.one_cycle ~max_value:1.0 ~total_steps:100 () in
              let v = S.eval s step in
              is_true ~msg:">=0" (v >= 0.0);
              is_true ~msg:"<=max" (v <= 1.0 +. 1e-6));
          prop "cosine_decay_restarts periodic" Gen.nat (fun step ->
              let period = 50 in
              let s =
                S.cosine_decay_restarts ~init_value:1.0 ~decay_steps:period ()
              in
              let v1 = S.eval s step in
              let v2 = S.eval s (step + period) in
              equal ~msg:"periodic" (float 1e-5) v1 v2);
        ];
      group "primitives"
        [
          test "scale" test_scale;
          test "scale_by_schedule" test_scale_by_schedule;
          test "scale_by_learning_rate" test_scale_by_learning_rate;
          test "trace" test_trace;
          test "trace nesterov" test_trace_nesterov;
          test "add_decayed_weights" test_add_decayed_weights;
          test "add_decayed_weights scheduled"
            test_add_decayed_weights_scheduled;
          test "clip" test_clip;
          test "clip_by_norm" test_clip_by_norm;
          test "clip_by_norm no-op" test_clip_by_norm_no_op;
          test "trust_ratio" test_trust_ratio;
          test "trust_ratio zero param" test_trust_ratio_zero_param;
          test "centralize 2d" test_centralize_2d;
          test "centralize 1d" test_centralize_1d;
          test "add_noise" test_add_noise;
        ];
      group "adam"
        [
          test "step 1 exact" test_scale_by_adam_step1;
          test "amsgrad holds max" test_amsgrad;
          test "nesterov differs" test_nesterov_differs;
        ];
      group "muon"
        [
          test "matches the reference Newton-Schulz iteration"
            test_muon_matches_reference_ns5;
          test "is transpose equivariant" test_muon_transpose_equivariant;
          test "orthogonalizes the update" test_muon_orthogonalizes;
          test "maps an isotropic spectrum through the quintic"
            test_muon_isotropic_spectrum;
          test "scaling conventions hold" test_muon_scale_conventions;
          test "momentum is an EMA, nesterov interpolates" test_muon_momentum;
          test "flattens convolutional filters"
            test_muon_flattens_higher_dimensions;
          test "converges on a matrix quadratic" test_muon_converges;
        ];
      group "optimizers"
        [
          test "lion converges" test_lion_converges;
          test "radam converges" test_radam_converges;
          test "adan converges" test_adan_converges;
          test "lamb converges" test_lamb_converges;
          test "lars converges" test_lars_converges;
          test "adafactor converges" test_adafactor_converges;
          test "adam+amsgrad converges" test_adam_amsgrad_converges;
        ];
      group "chain"
        [
          test "associativity" test_chain_associativity;
          test "identity" test_chain_identity;
          test "ordering matters" test_chain_ordering_matters;
        ];
      group "apply_if_finite"
        [
          test "finite passes through" test_finite_passes_through;
          test "nan skipped" test_nan_skipped;
          test "inf skipped" test_inf_skipped;
          test "counter tracks failures" test_nonfinite_counter;
        ];
      group "serialization"
        [
          test "n_tensors" test_n_tensors;
          test "round-trip" test_serialization_round_trip;
          test "wrong count raises" test_wrong_tensor_count;
        ];
      group "validation" [ test "invalid hyperparameters" test_validation ];
    ]
