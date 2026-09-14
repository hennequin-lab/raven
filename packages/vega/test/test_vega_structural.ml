(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tests for Vega's structural tier: optimizers over Nx.Ptree.S. *)

open Windtrap
module S = Vega.Schedule

(* A single float64 tensor, for analytic trajectory checks. *)
module Vec = struct
  type t = Nx.float64_t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t = f t
  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) = f
  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t = f t
end

(* Two float32 leaves of different shapes, for structural pairing checks. *)
module Pair = struct
  type t = { a : Nx.float32_t; b : Nx.float32_t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { a; b } =
    { a = f a; b = f b }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { a = f p.a q.a; b = f p.b q.b }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { a; b } =
    f a;
    f b
end

let vec xs = Nx.create Nx.float64 [| Array.length xs |] xs

let pair a b =
  {
    Pair.a = Nx.create Nx.float32 [| Array.length a |] a;
    b = Nx.create Nx.float32 [| Array.length b |] b;
  }

(* A matrix leaf and a vector leaf: Muon takes the matrix, the auxiliary AdamW
   the vector. *)
module Mixed = struct
  type t = { w : Nx.float32_t; b : Nx.float32_t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { w; b } =
    { w = f w; b = f b }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { w = f p.w q.w; b = f p.b q.b }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { w; b } =
    f w;
    f b
end

let mixed w b =
  {
    Mixed.w = Nx.create Nx.float32 [| 2; 3 |] w;
    b = Nx.create Nx.float32 [| Array.length b |] b;
  }

(* A leaf's elements as a flat host array, whatever its shape. *)
let flat t = Nx.to_array (Nx.reshape [| -1 |] (Nx.contiguous t))

let check_vec ?(eps = 1e-9) ?msg expected actual =
  let t = if eps = 0. then float_exact else float eps in
  equal ?msg (array t) expected (Nx.to_array actual)

(* The float64 analytic tests need the rate exact at float64; [Vega.lr] is
   float32 (cast up, it would perturb the last digits). The [~lr] argument
   takes any float dtype. *)
let lr64 v = Nx.scalar Nx.float64 v

(* Quadratic bowl over [Pair]: f p = ||p - target||^2, with analytic gradients,
   so tests exercise the optimizer alone. *)
let bowl_target = lazy (pair [| 1.5; -0.5 |] [| 2.0 |])
let bowl_start = lazy (pair [| 5.0; -3.0 |] [| -4.0 |])

let bowl_grads (params : Pair.t) =
  let target = Lazy.force bowl_target in
  {
    Pair.a = Nx.mul_s (Nx.sub params.a target.a) 2.0;
    b = Nx.mul_s (Nx.sub params.b target.b) 2.0;
  }

let bowl_distance params =
  Vega.global_norm
    (module Pair)
    (Pair.map2 Nx.sub params (Lazy.force bowl_target))

let descend ~steps ~step params =
  let rec loop k acc = if k = 0 then acc else loop (k - 1) (step acc) in
  loop steps params

(* Schedules. [S.eval] reads a schedule at a host counter; the values are
   float32, so expectations carry float32 tolerances. *)

let test_constant () =
  let sched = S.constant 0.1 in
  equal (float 1e-6) 0.1 (S.eval sched 0);
  equal (float 1e-6) 0.1 (S.eval sched 1000)

let test_exponential_decay () =
  let sched =
    S.exponential_decay ~init_value:0.5 ~decay_rate:0.1 ~decay_steps:100
  in
  equal (float 1e-6) 0.5 (S.eval sched 0);
  equal (float 1e-6) 0.05 (S.eval sched 100);
  equal (float 1e-6) 0.005 (S.eval sched 200)

let test_cosine_decay () =
  (* alpha = 0.1 makes the final value alpha * init_value = 0.01. *)
  let sched = S.cosine_decay ~init_value:0.1 ~decay_steps:100 ~alpha:0.1 () in
  equal (float 1e-6) 0.1 (S.eval sched 0);
  equal (float 1e-6) 0.055 (S.eval sched 50);
  equal (float 1e-6) 0.01 (S.eval sched 100);
  equal ~msg:"stays at final past steps" (float 1e-6) 0.01 (S.eval sched 250)

let test_warmup_cosine () =
  let sched =
    S.warmup_cosine_decay ~init_value:0.0 ~peak_value:1.0 ~warmup_steps:10
      ~decay_steps:100 ()
  in
  equal (float 1e-6) 0.0 (S.eval sched 0);
  equal (float 1e-6) 0.5 (S.eval sched 5);
  equal (float 1e-6) 1.0 (S.eval sched 10);
  equal ~msg:"cosine midpoint" (float 1e-6) 0.5 (S.eval sched 60);
  equal (float 1e-6) 0.0 (S.eval sched 110)

let test_schedule_validation () =
  raises
    (Invalid_argument "Schedule.exponential_decay: decay_steps must be positive")
    (fun () ->
      ignore
        (S.exponential_decay ~init_value:1.0 ~decay_rate:0.5 ~decay_steps:0
          : S.t));
  raises
    (Invalid_argument "Schedule.cosine_decay: decay_steps must be positive")
    (fun () -> ignore (S.cosine_decay ~init_value:1.0 ~decay_steps:(-1) () : S.t));
  raises
    (Invalid_argument
       "Schedule.warmup_cosine_decay: warmup_steps must be positive") (fun () ->
      ignore
        (S.warmup_cosine_decay ~init_value:0.0 ~peak_value:1.0 ~warmup_steps:0
           ~decay_steps:10 ()
          : S.t));
  raises
    (Invalid_argument
       "Schedule.warmup_cosine_decay: decay_steps must be positive") (fun () ->
      ignore
        (S.warmup_cosine_decay ~init_value:0.0 ~peak_value:1.0 ~warmup_steps:10
           ~decay_steps:0 ()
          : S.t))

(* Gradient transformations *)

let test_global_norm () =
  (* sqrt (3^2 + 0^2 + 4^2 + 12^2) = 13 *)
  let grads = pair [| 3.0; 0.0 |] [| 4.0; 12.0 |] in
  equal (float 1e-6) 13.0 (Vega.global_norm (module Pair) grads)

let test_clip_by_global_norm_rescales () =
  let grads = pair [| 3.0; 0.0 |] [| 4.0 |] in
  let clipped = Vega.clip_by_global_norm (module Pair) ~max_norm:1.0 grads in
  equal ~msg:"norm is the bound" (float 1e-6) 1.0
    (Vega.global_norm (module Pair) clipped);
  check_vec ~eps:1e-6 ~msg:"direction preserved" [| 0.6; 0.0 |] clipped.a;
  check_vec ~eps:1e-6 [| 0.8 |] clipped.b

let test_clip_by_global_norm_small () =
  let grads = pair [| 3.0; 0.0 |] [| 4.0 |] in
  let clipped = Vega.clip_by_global_norm (module Pair) ~max_norm:10.0 grads in
  check_vec ~eps:0. [| 3.0; 0.0 |] clipped.a;
  check_vec ~eps:0. [| 4.0 |] clipped.b;
  let zeros = pair [| 0.0; 0.0 |] [| 0.0 |] in
  let clipped = Vega.clip_by_global_norm (module Pair) ~max_norm:1.0 zeros in
  check_vec ~eps:0. ~msg:"zero gradients pass through" [| 0.0 |] clipped.b

let test_clip_by_value () =
  let grads = pair [| -3.0; 0.2 |] [| 5.0 |] in
  let clipped = Vega.clip_by_value (module Pair) ~max:1.0 grads in
  check_vec ~eps:1e-7 [| -1.0; 0.2 |] clipped.a;
  check_vec ~eps:0. [| 1.0 |] clipped.b

let test_clip_validation () =
  let grads = pair [| 1.0 |] [| 1.0 |] in
  raises
    (Invalid_argument "Vega.clip_by_global_norm: expected max_norm > 0.0, got 0")
    (fun () -> Vega.clip_by_global_norm (module Pair) ~max_norm:0.0 grads);
  raises (Invalid_argument "Vega.clip_by_value: expected max > 0.0, got -1")
    (fun () -> Vega.clip_by_value (module Pair) ~max:(-1.0) grads)

(* SGD *)

let test_sgd_first_step () =
  let params = vec [| 1.0; -2.0 |] in
  let grads = vec [| 0.5; -1.0 |] in
  let st = Vega.sgd_init (module Vec) params in
  (* Zero velocity: the first step is plain descent even with momentum. *)
  let params', st' =
    Vega.sgd_step (module Vec) ~lr:(lr64 0.1) ~momentum:0.9 st ~params ~grads
  in
  check_vec [| 0.95; -1.9 |] params';
  check_vec ~msg:"velocity is the gradient" [| 0.5; -1.0 |] st'.velocity

let test_sgd_velocity_threads () =
  let params = vec [| 0.0 |] in
  let st = Vega.sgd_init (module Vec) params in
  let params, st =
    Vega.sgd_step
      (module Vec)
      ~lr:(lr64 0.1) ~momentum:0.5 st ~params ~grads:(vec [| 1.0 |])
  in
  let _, st =
    Vega.sgd_step
      (module Vec)
      ~lr:(lr64 0.1) ~momentum:0.5 st ~params ~grads:(vec [| 2.0 |])
  in
  (* v2 = 0.5 *. v1 +. g2 = 0.5 *. 1. +. 2. *)
  check_vec [| 2.5 |] st.velocity;
  equal ~msg:"counter reads 2" int 2 (Int32.to_int (Nx.item [] st.step))

let test_sgd_converges () =
  let params = Lazy.force bowl_start in
  let step (params, st) =
    let grads = bowl_grads params in
    Vega.sgd_step (module Pair) ~lr:(Vega.lr 0.1) st ~params ~grads
  in
  let params, _ =
    descend ~steps:100 ~step (params, Vega.sgd_init (module Pair) params)
  in
  is_true ~msg:"reaches the bottom of the bowl" (bowl_distance params < 1e-3)

let test_sgd_momentum_converges () =
  let params = Lazy.force bowl_start in
  let step (params, st) =
    let grads = bowl_grads params in
    Vega.sgd_step
      (module Pair)
      ~lr:(Vega.lr 0.05) ~momentum:0.9 st ~params ~grads
  in
  let params, _ =
    descend ~steps:200 ~step (params, Vega.sgd_init (module Pair) params)
  in
  is_true ~msg:"reaches the bottom of the bowl" (bowl_distance params < 1e-3)

let test_sgd_pairs_leaves_structurally () =
  let params = pair [| 1.0; 2.0 |] [| 3.0 |] in
  let grads = pair [| 0.0; 0.0 |] [| 1.0 |] in
  let st = Vega.sgd_init (module Pair) params in
  let params', _ =
    Vega.sgd_step (module Pair) ~lr:(Vega.lr 0.5) st ~params ~grads
  in
  check_vec ~eps:0. ~msg:"zero-gradient leaf untouched" [| 1.0; 2.0 |] params'.a;
  check_vec ~eps:0. [| 2.5 |] params'.b

(* Adam *)

let test_adam_first_step () =
  let b1 = 0.9 and b2 = 0.999 and eps = 1e-8 and lr = 0.1 in
  let g = [| 4.0; -0.5; 0.0 |] in
  let params = vec [| 1.0; -2.0; 3.0 |] in
  let st = Vega.adam_init (module Vec) params in
  let params', st' =
    Vega.adam_step (module Vec) ~lr:(lr64 lr) st ~params ~grads:(vec g)
  in
  (* First step analytically: mu = (1-b1) g, nu = (1-b2) g^2, and the
     bias-corrected direction is g / (|g| + eps). *)
  let expected =
    Array.map2
      (fun p g -> p -. (lr *. g /. (Float.abs g +. eps)))
      (Nx.to_array params) g
  in
  check_vec expected params';
  check_vec ~msg:"mu" (Array.map (fun g -> (1. -. b1) *. g) g) st'.mu;
  check_vec ~msg:"nu" (Array.map (fun g -> (1. -. b2) *. g *. g) g) st'.nu;
  equal ~msg:"step count" int 1 (Int32.to_int (Nx.item [] st'.step))

let test_adam_reference_trajectory () =
  let b1 = 0.9 and b2 = 0.999 and eps = 1e-8 and lr = 0.05 in
  let grad p = 2.0 *. (p -. 1.0) in
  (* Scalar reference implementation in plain floats. *)
  let expected =
    let p = ref 3.0 and mu = ref 0.0 and nu = ref 0.0 in
    List.init 10 (fun i ->
        let t = i + 1 in
        let g = grad !p in
        mu := (b1 *. !mu) +. ((1.0 -. b1) *. g);
        nu := (b2 *. !nu) +. ((1.0 -. b2) *. g *. g);
        let mu_hat = !mu /. (1.0 -. (b1 ** float_of_int t)) in
        let nu_hat = !nu /. (1.0 -. (b2 ** float_of_int t)) in
        p := !p -. (lr *. mu_hat /. (Stdlib.sqrt nu_hat +. eps));
        !p)
  in
  let params = ref (vec [| 3.0 |]) in
  let st = ref (Vega.adam_init (module Vec) !params) in
  List.iteri
    (fun i e ->
      let grads = Nx.mul_s (Nx.sub_s !params 1.0) 2.0 in
      let params', st' =
        Vega.adam_step (module Vec) ~lr:(lr64 lr) !st ~params:!params ~grads
      in
      params := params';
      st := st';
      check_vec ~msg:(Printf.sprintf "step %d" (i + 1)) [| e |] !params)
    expected

let test_adam_converges () =
  let params = Lazy.force bowl_start in
  let step (params, st) =
    let grads = bowl_grads params in
    Vega.adam_step (module Pair) ~lr:(Vega.lr 0.02) st ~params ~grads
  in
  let params, _ =
    descend ~steps:800 ~step (params, Vega.adam_init (module Pair) params)
  in
  is_true ~msg:"reaches the bottom of the bowl" (bowl_distance params < 0.05)

let test_adam_with_schedule_converges () =
  (* The learning rate comes from the state's own step counter through the
     schedule — the jitted loop's shape, run eagerly. *)
  let sched = S.cosine_decay ~init_value:0.1 ~decay_steps:300 () in
  let params = Lazy.force bowl_start in
  let state = ref (params, Vega.adam_init (module Pair) params) in
  for _k = 1 to 300 do
    let params, st = !state in
    let grads = bowl_grads params in
    state :=
      Vega.adam_step (module Pair) ~lr:(sched st.step) st ~params ~grads
  done;
  is_true ~msg:"decayed steps settle at the bottom"
    (bowl_distance (fst !state) < 0.02)

let test_adam_zero_grads () =
  let params = vec [| 1.0; -2.0 |] in
  let st = Vega.adam_init (module Vec) params in
  let params', st' =
    Vega.adam_step
      (module Vec)
      ~lr:(Vega.lr 0.1) st ~params
      ~grads:(vec [| 0.0; 0.0 |])
  in
  check_vec ~eps:0. ~msg:"parameters unchanged" [| 1.0; -2.0 |] params';
  equal ~msg:"step still advances" int 1 (Int32.to_int (Nx.item [] st'.step))

let test_adam_step_is_pure () =
  let params = vec [| 3.0; -1.0 |] in
  let grads = vec [| 0.7; 0.3 |] in
  let st = Vega.adam_init (module Vec) params in
  let once, _ =
    Vega.adam_step (module Vec) ~lr:(Vega.lr 0.1) st ~params ~grads
  in
  let again, _ =
    Vega.adam_step (module Vec) ~lr:(Vega.lr 0.1) st ~params ~grads
  in
  check_vec ~eps:0. ~msg:"same state, same step" (Nx.to_array once) again

(* AdamW *)

let test_adamw_zero_decay_is_adam () =
  let grads_of params = Nx.mul_s (Nx.sub_s params 1.0) 2.0 in
  let run step =
    let params = ref (vec [| 3.0; -2.0 |]) in
    let st = ref (Vega.adam_init (module Vec) !params) in
    for _ = 1 to 5 do
      let params', st' = step !st ~params:!params ~grads:(grads_of !params) in
      params := params';
      st := st'
    done;
    !params
  in
  let adam =
    run (fun st ~params ~grads ->
        Vega.adam_step (module Vec) ~lr:(Vega.lr 0.1) st ~params ~grads)
  in
  let adamw =
    run (fun st ~params ~grads ->
        Vega.adamw_step
          (module Vec)
          ~lr:(Vega.lr 0.1) ~weight_decay:0.0 st ~params ~grads)
  in
  check_vec ~eps:0. (Nx.to_array adam) adamw

let test_adamw_decays_weights () =
  (* Zero gradients isolate the decay: p_k = p_0 (1 - lr wd)^k. A coupled (L2)
     decay would instead be distorted by the adaptive scaling. *)
  let lr = 0.1 and wd = 0.5 in
  let p0 = [| 2.0; -4.0 |] in
  let params = ref (vec p0) in
  let st = ref (Vega.adamw_init (module Vec) !params) in
  for _ = 1 to 3 do
    let params', st' =
      Vega.adamw_step
        (module Vec)
        ~lr:(lr64 lr) ~weight_decay:wd !st ~params:!params
        ~grads:(vec [| 0.0; 0.0 |])
    in
    params := params';
    st := st'
  done;
  let c = (1.0 -. (lr *. wd)) ** 3.0 in
  check_vec (Array.map (fun p -> p *. c) p0) !params

let test_adamw_converges () =
  let params = Lazy.force bowl_start in
  let step (params, st) =
    let grads = bowl_grads params in
    Vega.adamw_step
      (module Pair)
      ~lr:(Vega.lr 0.02) ~weight_decay:1e-3 st ~params ~grads
  in
  let params, _ =
    descend ~steps:800 ~step (params, Vega.adamw_init (module Pair) params)
  in
  is_true ~msg:"reaches the bottom of the bowl" (bowl_distance params < 0.05)

(* Muon *)

let test_muon_matches_the_chain () =
  (* On a matrix-only parameter tree the structural step is the per-tensor Muon
     transform and its update: the same orthogonalized momentum, the same shape
     factor, the same decoupled decay. *)
  let lr = 0.0625 and wd = 0.01 in
  let tensor xs = Nx.create Nx.float64 [| 3; 3 |] xs in
  let param0 = tensor [| 0.5; -1.0; 2.0; 0.25; 1.5; -0.5; -2.0; 0.75; 1.0 |] in
  let grads =
    List.map tensor
      [
        [| 1.0; 0.5; -0.25; 0.75; -1.0; 2.0; 0.5; -0.5; 1.25 |];
        [| -0.5; 1.0; 0.25; 1.5; 0.75; -1.25; 0.25; 0.5; -0.75 |];
      ]
  in
  let tx =
    Vega.chain
      [
        Vega.scale_by_muon ~momentum:0.9 ~nesterov:true
          ~scaling:(`Update_rms 0.2) ();
        Vega.add_decayed_weights ~rate:(S.constant wd) ();
        Vega.scale_by_learning_rate (S.constant lr);
      ]
  in
  let chain_params, _ =
    List.fold_left
      (fun (param, st) grad ->
        let updates, st = Vega.update st ~grad ~param in
        (Vega.apply_updates ~param ~updates, st))
      (param0, Vega.init tx param0)
      grads
  in
  let step_params, _ =
    List.fold_left
      (fun (param, st) grad ->
        Vega.muon_step
          (module Vec)
          ~lr:(lr64 lr) ~aux_lr:(lr64 0.001) ~momentum:0.9 ~nesterov:true
          ~scaling:(`Update_rms 0.2) st ~params:param ~grads:grad)
      (param0, Vega.muon_init (module Vec) param0)
      grads
  in
  equal ~msg:"the matrix path matches the chain"
    (array (float 1e-6))
    (flat chain_params) (flat step_params)

let test_muon_aux_is_adamw () =
  (* Every leaf the routing rule does not send to Muon must be optimized by
     exactly [adamw_step], weight decay and all: routing the whole structure to
     the auxiliary optimizer reproduces AdamW. *)
  let params0 = mixed [| 1.0; -2.0; 3.0; 0.5; -1.5; 2.0 |] [| 0.25; -0.75 |] in
  let grads =
    [
      mixed [| 0.5; 0.25; -1.0; 0.75; 0.5; -0.25 |] [| 0.1; -0.2 |];
      mixed [| -0.25; 0.5; 0.75; -0.5; 1.0; 0.25 |] [| -0.3; 0.4 |];
    ]
  in
  let aux_only (_ : int array) = false in
  let muon_params, _ =
    List.fold_left
      (fun (params, st) grads ->
        Vega.muon_step
          (module Mixed)
          ~lr:(Vega.lr 0.1) ~aux_lr:(Vega.lr 0.1) ~use_muon:aux_only st ~params
          ~grads)
      (params0, Vega.muon_init (module Mixed) ~use_muon:aux_only params0)
      grads
  in
  let adamw_params, _ =
    List.fold_left
      (fun (params, st) grads ->
        Vega.adamw_step (module Mixed) ~lr:(Vega.lr 0.1) st ~params ~grads)
      (params0, Vega.adamw_init (module Mixed) params0)
      grads
  in
  equal ~msg:"the matrix leaf is AdamW's" (array float_exact)
    (flat adamw_params.Mixed.w)
    (flat muon_params.Mixed.w);
  equal ~msg:"the vector leaf is AdamW's" (array float_exact)
    (flat adamw_params.Mixed.b)
    (flat muon_params.Mixed.b)

let test_muon_routes_and_allocates () =
  (* The default rule routes by shape, and a buffer is allocated only on the
     leaves that use it: the matrix on Muon carries a momentum buffer and
     zero-dimensional placeholders for the AdamW moments, the vector leaf the
     other way round. *)
  let params = mixed [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] [| 0.0 |] in
  let grads = mixed [| 0.1; 0.2; 0.3; 0.4; 0.5; 0.6 |] [| 0.7 |] in
  let st = Vega.muon_init (module Mixed) params in
  equal ~msg:"velocity on the matrix" (array int) [| 2; 3 |]
    (Nx.shape st.velocity.w);
  equal ~msg:"no velocity on the vector" (array int) [||]
    (Nx.shape st.velocity.b);
  equal ~msg:"no moments on the matrix" (array int) [||] (Nx.shape st.aux_mu.w);
  equal ~msg:"moments on the vector" (array int) [| 1 |] (Nx.shape st.aux_mu.b);
  equal ~msg:"second moment too" (array int) [| 1 |] (Nx.shape st.aux_nu.b);
  let _, st =
    Vega.muon_step
      (module Mixed)
      ~lr:(Vega.lr 0.1) ~aux_lr:(Vega.lr 0.1) st ~params ~grads
  in
  (* The matrix leaf's momentum is the EMA of its gradient (momentum 0.95); the
     vector leaf's first moment is AdamW's, [0.1 * g]. *)
  equal ~msg:"matrix momentum"
    (array (float 1e-6))
    (Array.map (fun g -> 0.05 *. g) [| 0.1; 0.2; 0.3; 0.4; 0.5; 0.6 |])
    (flat st.velocity.w);
  equal ~msg:"vector moment" (array (float 1e-6)) [| 0.07 |] (flat st.aux_mu.b)

let test_muon_zero_grads () =
  let params = mixed [| 1.0; -2.0; 3.0; 0.5; -1.5; 2.0 |] [| 0.25 |] in
  let grads = mixed (Array.make 6 0.0) [| 0.0 |] in
  let st = Vega.muon_init (module Mixed) params in
  let params', st' =
    Vega.muon_step
      (module Mixed)
      ~lr:(Vega.lr 0.1) ~aux_lr:(Vega.lr 0.1) ~weight_decay:0.0 st ~params
      ~grads
  in
  equal ~msg:"the matrix leaf is unchanged" (array float_exact) (flat params.w)
    (flat params'.w);
  equal ~msg:"the vector leaf is unchanged" (array float_exact) (flat params.b)
    (flat params'.b);
  (* Orthogonalizing a zero matrix is zero, not a division by zero. *)
  is_true ~msg:"the momentum stays finite"
    (Nx.item [] (Nx.all (Nx.isfinite st'.velocity.w)));
  equal ~msg:"the counter advances" int 1 (Int32.to_int (Nx.item [] st'.step))

let test_muon_rejects_non_matrix_routing () =
  (* Muon is a matrix method, so a routing rule that selects a vector leaf is an
     error when the state is built, not a silently wrong update. *)
  raises_match Exn.invalid_arg (fun () ->
      ignore
        (Vega.muon_init
           (module Pair)
           ~use_muon:(fun _ -> true)
           (pair [| 1.0 |] [| 2.0 |])));
  raises_match Exn.invalid_arg (fun () ->
      let params = pair [| 1.0 |] [| 2.0 |] in
      ignore
        (Vega.muon_step
           (module Pair)
           ~lr:(Vega.lr 0.1) ~aux_lr:(Vega.lr 0.1) ~momentum:1.0
           (Vega.muon_init (module Pair) params)
           ~params ~grads:params))

let test_muon_state_traversals () =
  (* The Muon state is a parameter tree like the others: one traversal visits
     the matrix leaf's three buffers, the vector leaf's three, then the
     counter. *)
  let module M = Vega.Muon_state (Mixed) in
  let params = mixed (Array.make 6 1.0) [| 1.0 |] in
  let st = Vega.muon_init (module Mixed) params in
  let n = ref 0 in
  M.iter (fun _ -> incr n) st;
  equal ~msg:"muon leaf count" int 7 !n;
  equal ~msg:"the counter starts at zero" int 0
    (Int32.to_int (Nx.item [] st.step));
  let merged = M.map2 (fun _ right -> right) st st in
  equal ~msg:"map2 merges leafwise" int32 0l (Nx.item [] merged.step);
  equal ~msg:"placeholders merge too" (array int) [||]
    (Nx.shape merged.velocity.b)

(* A parameter structure may carry leaves that are not parameters. The canonical
   one is an RNG key, which has to sit in the structure to reach a compiled step
   as an input but is not something to optimize. Rune leaves its gradient slot
   at zero; the optimizers must leave the value alone. Adam is where it would
   show: its direction runs each leaf through a square root and a division. *)
module Stepper = struct
  type t = { w : Nx.float64_t; key : Nx.Rng.key }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t =
    { w = f t.w; key = f t.key }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    { w = f a.w b.w; key = f a.key b.key }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t =
    f t.w;
    f t.key
end

let test_optimizers_carry_a_non_parameter_leaf () =
  let params =
    Stepper.
      {
        w = Nx.create Nx.float64 [| 3 |] [| 1.0; -2.0; 3.0 |];
        key = Nx.Rng.key 7;
      }
  in
  let grads =
    Stepper.
      {
        w = Nx.create Nx.float64 [| 3 |] [| 2.0; -4.0; 6.0 |];
        key = Nx.zeros Nx.int32 [| 2 |];
      }
  in
  let key_before = Nx.to_array params.Stepper.key in
  let check name (updated : Stepper.t) =
    is_true
      ~msg:(name ^ " moves the weight")
      (Nx.to_array updated.Stepper.w <> Nx.to_array params.Stepper.w);
    equal
      ~msg:(name ^ " leaves the key alone")
      (array int32) key_before
      (Nx.to_array updated.Stepper.key)
  in
  let sgd, _ =
    Vega.sgd_step
      (module Stepper)
      ~lr:(Vega.lr 0.1)
      (Vega.sgd_init (module Stepper) params)
      ~params ~grads
  in
  check "sgd" sgd;
  let momentum, _ =
    Vega.sgd_step
      (module Stepper)
      ~lr:(Vega.lr 0.1) ~momentum:0.9
      (Vega.sgd_init (module Stepper) params)
      ~params ~grads
  in
  check "sgd with momentum" momentum;
  let adam, _ =
    Vega.adam_step
      (module Stepper)
      ~lr:(Vega.lr 0.1)
      (Vega.adam_init (module Stepper) params)
      ~params ~grads
  in
  check "adam" adam;
  let adamw, _ =
    Vega.adamw_step
      (module Stepper)
      ~lr:(Vega.lr 0.1)
      (Vega.adamw_init (module Stepper) params)
      ~params ~grads
  in
  check "adamw" adamw;
  let muon, _ =
    Vega.muon_step
      (module Stepper)
      ~lr:(Vega.lr 0.1) ~aux_lr:(Vega.lr 0.01)
      (Vega.muon_init (module Stepper) params)
      ~params ~grads
  in
  check "muon" muon

(* Optimizer state as a parameter tree *)

let test_adam_counter_advances () =
  let params = vec [| 1.0 |] in
  let grads = vec [| 1.0 |] in
  let st = ref (Vega.adam_init (module Vec) params) in
  let lr = Vega.lr 0.1 in
  (* The counter is a tensor leaf, so it advances through the state alone —
     the shape a compiled loop relies on. The bias corrections derive from it
     inside each step (checked against the closed form by the reference
     trajectory above). *)
  for _ = 1 to 5 do
    let _, st' = Vega.adam_step (module Vec) ~lr !st ~params ~grads in
    st := st'
  done;
  equal ~msg:"counter reads 5 after 5 steps" int 5
    (Int32.to_int (Nx.item [] !st.step))

let test_state_traversals () =
  (* Both state functors are parameter trees: map/map2/iter walk every tensor
     leaf — payload leaves, then the counter — in a fixed order. *)
  let module A = Vega.Adam_state (Pair) in
  let module Sg = Vega.Sgd_state (Pair) in
  let double (type a b) (t : (a, b) Nx.t) : (a, b) Nx.t =
    Nx.cast (Nx.dtype t) (Nx.mul_s (Nx.cast Nx.float64 t) 2.0)
  in
  let params = pair [| 1.0; 2.0 |] [| 3.0 |] in
  let st = Vega.adam_init (module Pair) params in
  (* map doubles everything; iter counts the leaves it visits. *)
  let doubled = A.map double st in
  check_vec ~msg:"mu doubled" [| 0.0; 0.0 |] doubled.mu.a;
  let n = ref 0 in
  A.iter (fun _ -> incr n) st;
  (* 2 mu leaves + 2 nu leaves + step. *)
  equal ~msg:"adam leaf count" int 5 !n;
  (* map2 merges leafwise: take the right state everywhere. *)
  let st' = A.map2 (fun _ r -> r) st doubled in
  check_vec ~msg:"merged mu" [| 0.0 |] st'.mu.b;
  equal ~msg:"merged step" int32 0l (Nx.item [] st'.step);
  (* The sgd state: 2 velocity leaves + step. *)
  let sst = Vega.sgd_init (module Pair) params in
  let n = ref 0 in
  Sg.iter (fun _ -> incr n) sst;
  equal ~msg:"sgd leaf count" int 3 !n

let test_state_functor_is_a_ptree () =
  (* [Vega.Adam_state (P)] is an Nx.Ptree.S: the state can sit inside another
     tree — the shape a jitted step's input record takes. *)
  let module Opt = Vega.Adam_state (Pair) in
  let params = pair [| 1.0 |] [| 2.0 |] in
  let grads = pair [| 0.5 |] [| -0.5 |] in
  let st = Vega.adam_init (module Pair) params in
  (* Run the step through the state's own walker: embedding the state in an
     outer record and mapping over it must reproduce the state exactly. *)
  let roundtrip =
    Opt.map (fun t -> t) (Opt.map2 (fun _ r -> r) (Opt.map (fun t -> t) st) st)
  in
  let params', _ =
    Vega.adam_step (module Pair) ~lr:(Vega.lr 0.1) roundtrip ~params ~grads
  in
  let expected, _ =
    Vega.adam_step (module Pair) ~lr:(Vega.lr 0.1) st ~params ~grads
  in
  check_vec ~msg:"roundtrip state steps identically" (Nx.to_array expected.a)
    params'.a

let tests =
  [
    group "schedules"
      [
        test "constant is constant" test_constant;
        test "exponential decay is geometric in steps" test_exponential_decay;
        test "cosine decay spans init to final" test_cosine_decay;
        test "warmup cosine ramps then decays" test_warmup_cosine;
        test "constructors reject bad step counts" test_schedule_validation;
      ];
    group "gradient transformations"
      [
        test "global norm spans all leaves" test_global_norm;
        test "clip by global norm rescales to the bound"
          test_clip_by_global_norm_rescales;
        test "clip by global norm passes small gradients through"
          test_clip_by_global_norm_small;
        test "clip by value clamps elementwise" test_clip_by_value;
        test "clipping rejects non-positive bounds" test_clip_validation;
      ];
    group "sgd"
      [
        test "first step is plain gradient descent" test_sgd_first_step;
        test "velocity threads across steps" test_sgd_velocity_threads;
        test "converges on a quadratic bowl" test_sgd_converges;
        test "momentum converges on a quadratic bowl"
          test_sgd_momentum_converges;
        test "pairs leaves structurally, not positionally"
          test_sgd_pairs_leaves_structurally;
      ];
    group "adam"
      [
        test "first step matches the analytic update" test_adam_first_step;
        test "follows the scalar reference trajectory"
          test_adam_reference_trajectory;
        test "converges on a quadratic bowl" test_adam_converges;
        test "converges under a cosine schedule"
          test_adam_with_schedule_converges;
        test "the counter advances as a tensor" test_adam_counter_advances;
        test "zero gradients leave parameters unchanged" test_adam_zero_grads;
        test "stepping is pure in the threaded state" test_adam_step_is_pure;
      ];
    group "adamw"
      [
        test "zero weight decay reduces to adam" test_adamw_zero_decay_is_adam;
        test "zero gradients decay weights geometrically"
          test_adamw_decays_weights;
        test "converges on a quadratic bowl" test_adamw_converges;
      ];
    group "muon"
      [
        test "the matrix path matches the per-tensor chain"
          test_muon_matches_the_chain;
        test "the auxiliary optimizer is AdamW" test_muon_aux_is_adamw;
        test "routes by shape and allocates buffers"
          test_muon_routes_and_allocates;
        test "zero gradients leave parameters unchanged" test_muon_zero_grads;
        test "rejects a non-matrix routing rule"
          test_muon_rejects_non_matrix_routing;
        test "state traversals walk every leaf" test_muon_state_traversals;
      ];
    group "optimizer state as a parameter tree"
      [
        test "state traversals walk every leaf" test_state_traversals;
        test "the state functor is a Ptree.S that steps identically"
          test_state_functor_is_a_ptree;
      ];
    group "non-parameter leaves"
      [
        test "every optimizer carries them unchanged"
          test_optimizers_carry_a_non_parameter_leaf;
      ];
  ]

let () = run "vega structural" tests
