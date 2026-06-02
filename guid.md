Phase 1: Mathematical Stability (FP32 Fixes)

Objective: Eliminate the North Pole blowout and the event horizon pixelation
using Variable Transformation and Kahan Summation, allowing the kernel to
remain 100% in fast ti.f32.

- [ ] 1.1. Precompute Horizon Constants in Python
    - Where: taichi_renderer.py (render_beauty_frame host function)
    - Action: Calculate the event horizon distance before launching the kernel.
    - Code:
      k_horizon = math.sqrt(1.0 - a * a)
      r_plus = 1.0 + k_horizon
      # Pass k_horizon and r_plus into the render_beauty kernel
- [ ] 1.2. Rewrite _delta for Perfect FP32 Precision (Variable Transform 1)
    - Where: taichi_renderer.py (_delta)
    - Action: Replace the r^2 - 2r + a^2 formula. Create a new function that
      takes y (where y = r - r_+).
    - Code:
      @ti.func
      def _delta_y(y, k):
          return y * (y + 2.0 * k) # Zero catastrophic cancellation
- [ ] 1.3. Replace \theta with u = \cos\theta (Variable Transform 2)
    - Where: taichi_renderer.py (State vector and _deriv / _project functions)
    - Action: Change the state vector from [r, theta, phi, t, v_r, v_theta] to
      [y, u, phi, t, v_y, v_u].
    - Detail:
        - Initialization: y = r_cam - r_plus and u = ti.cos(theta_cam).
        - In _deriv, recover r: r = y + r_plus.
        - Replace ti.sin(theta)**2 with sin2 = 1.0 - u * u.
        - CRITICAL: Remove _SIN2_MIN and ti.max(sin2, _SIN2_MIN). The
          singularity is mathematically gone.
- [ ] 1.4. Apply Kahan Summation to the Integration State
    - Where: taichi_renderer.py (_rk4_step and the while loop)
    - Action: When adding the RK4 delta to your state vector (and when
      accumulating ray_length), use Compensated Summation to prevent drift.
    - Code:
      # Add a state compensation vector `c_sp` initialized to 0
      y_step = delta_sp - c_sp
      t_sp = sp + y_step
      c_sp = (t_sp - sp) - y_step
      sp = t_sp

Phase 2: Performance Overhaul (Hitting the Time Budget)

Objective: Cut the physics workload by >50%, utilize GPU hardware texture units,
and avoid over-calculating flat space.

- [ ] 2.1. Delete the Offset Ray & Shared Loop
    - Where: taichi_renderer.py (render_pipe_a and render_beauty)
    - Action: Remove so (the offset ray), Eo, Lo, Qo, and all conditions
      checking out_o. Trace only sp (the primary ray) in the while loop.
- [ ] 2.2. Implement "Smart" Adaptive Step Sizing
    - Where: taichi_renderer.py (inside the while step < n_steps loop)
    - Action: Scale the step size dynamically based on distance to the horizon.
    - Code:
      # y is sp[0]. It approaches 0 at the horizon.
      local_h = d_lambda * ti.max(0.005, y / (y + 2.0)) 
      sp = _rk4_step(sp, Ep, Lp, Qp, a, k_horizon, r_plus, local_h)
- [ ] 2.3. Replace Manual Mipmapping with Hardware ti.Texture
    - Where: taichi_renderer.py (setup_renderer)
    - Action: Delete star_flat, star_off, star_w, star_h, and _sample_trilinear.
    - Code:
      # In setup_renderer (Python side):
      starmap_tex = ti.Texture(ti.Format.rgba16f, (w, h))
      starmap_tex.from_numpy(starmap_numpy_array)
- [ ] 2.4. Implement Screen-Space Jacobian (The Shading Pass)
    - Where: taichi_renderer.py
    - Action: Split your giant kernel into two consecutive kernels.
        - Kernel 1 (Physics): Writes the final phi_exit and u_exit to a 2D
          ti.field(shape=(height, width, 2)).
        - Kernel 2 (Shading): Reads the field. Calculates the LOD by comparing
          [py, px] with its neighbors [py, px+1] and [py+1, px].
        - Texture Lookup: color = starmap_tex.sample_lod(vec2(u_tex, v_tex),
          calculated_lod)

Phase 3: Volumetrics & Deep Compositing Setup

Objective: Export perfect Z-Depth for Blender and perfectly occlude the
spaceship without aliased edges.

- [ ] 3.1. Feed Blender Spaceship Depth into Taichi
    - Where: taichi_renderer.py (Arguments to render_beauty_frame)
    - Action: Create a new ti.Texture(ti.Format.r32f, shape=(width, height)) to
      hold the Spaceship's Z-Depth pass rendered in Phase 1. Pass this to the
      kernel.
- [ ] 3.2. Implement Early Ray Termination (Spatial Occlusion)
    - Where: taichi_renderer.py (Inside the while loop of Kernel 1)
    - Action: Stop calculating physics if the gas is behind the ship.
    - Code:
      ship_z = ship_depth_tex.fetch(ti.Vector([px, py]), 0).x
      if ray_length > ship_z:
          out_p = _OCCLUDED_BY_SHIP
          break
- [ ] 3.3. Add Bounding Box Early-Outs for the Disk
    - Where: taichi_renderer.py (render_beauty before _disk_emit)
    - Action: Skip expensive 4-velocity math if the ray is far from the disk.
    - Code: if r < r_inner or r > r_outer or ti.abs(u) > sin_theta_half:
      continue
- [ ] 3.4. Calculate Transmittance-Weighted Z-Depth
    - Where: taichi_renderer.py (Disk emission block)
    - Action: Create depth_pixels = ti.field(dtype=ti.f32, shape=(height,
      width)).
    - Code:
      contribution = transm * emission_intensity
      weighted_depth += ray_length * contribution
      total_emission += contribution

      # After the while loop finishes:
      if total_emission > 1e-6:
          depth_pixels[py, px] = weighted_depth / total_emission
      else:
          depth_pixels[py, px] = 1e5 # Infinity

Phase 4: 360° Camera & Motion Blur

Objective: Correct the math for Equirectangular VR output and relativistic
motion blur.

- [ ] 4.1. Rewrite Ray Generation for 360° Equirectangular
    - Where: taichi_renderer.py (Beginning of render_beauty kernel)
    - Action: Remove tan_half_fov perspective math.
    - Code:
      lon = (px / width) * 2.0 * math.pi       # Azimuth
      lat = (py / height) * math.pi            # Elevation

      # Map to Cartesian forward/right/up local vectors
      npr_r = ti.sin(lat) * ti.cos(lon) # Example mapping
      npr_th = ti.cos(lat)
      npr_ph = ti.sin(lat) * ti.sin(lon)
      # Normalize and feed to _zamo_init
- [ ] 4.2. Implement Temporal Motion Blur (Inside Taichi)
    - Where: taichi_renderer.py (Outer structure of Kernel 1)
    - Action: Do NOT try to output Velocity EXRs for Blender's Vector Blur node.
    - Detail: Wrap the physics code for each pixel in a small loop for sample in
      range(4):. Jitter the starting phi_cam and theta_cam slightly based on the
      camera's rotational velocity over the 1/48th shutter interval. Average
      the 4 samples into frame_pixels.

Phase 5: Multi-Channel EXR Export Pipeline

Objective: Wrap the Taichi outputs into a standard G-Buffer format for Blender.

- [ ] 5.1. Extract Arrays from Taichi
    - Where: gpu_test.py or your main rendering script.
    - Action: Sync and export beauty_rgb = frame_pixels.to_numpy() and depth_z =
      depth_pixels.to_numpy().
- [ ] 5.2. Write Multi-Channel EXR via OpenImageIO
    - Where: scripts/ (Pipeline export script)
    - Action: Stack the numpy arrays and assign channel names.
    - Code:
      import OpenImageIO as oiio
      import numpy as np

      # Expand depth_z to have 3rd dimension if it's 2D
      depth_z = np.expand_dims(depth_z, axis=2)
      pixels = np.concatenate([beauty_rgb, depth_z], axis=2)

      out = oiio.ImageOutput.create(filename)
      spec = oiio.ImageSpec(width, height, 4, oiio.FLOAT)
      spec.channelnames = ("R", "G", "B", "Z")

      out.open(filename, spec)
      out.write_image(pixels)
      out.close()

Summary of Execution Order

1.  Change the Data Structures first: Implement y, u, and change vec6 to use
    them. Fix the potentials.
2.  Setup the Kernel Split: Move the texture/LOD code into a secondary Shading
    Kernel and remove the offset ray.
3.  Add the Features: Integrate the Kahan summation, adaptive d_lambda, and
    Equirectangular camera mapping.
4.  Finalize I/O: Add the Ship Depth input, the Z-Depth output, and the OIIO EXR
    export script.

By executing this checklist, you will bypass FP64 limits entirely, eliminate
visual artifacts at both the pole and horizon, perfectly depth-composite your
spaceships, and slash render times to your exact target.