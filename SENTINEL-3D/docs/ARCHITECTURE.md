# Data Flow & Tech Stack Architecture

The **Kavach 2.5D Ecosystem** is uniquely designed to decouple heavy AI edge-compute from the client visualisation. 

Rather than sending gigabytes of 3D point cloud data (.bin / .pcd) over the wire and parsing it in JavaScript (which crashes web browsers instantly), all major lifting is kept on the PyTorch backend. The node client only ever receives an ultra-compressed payload of JSON polygons, giving the illusion of a full point cloud while using `< 0.1%` of the bandwidth and memory.

---

## 1. The Technology Stack

### Backend (Execution Environment)
- **Language**: Python 3.10+
- **Deep Learning**: PyTorch 2.x
- **Sparse Convolutions**: `spconv v2` (hardware-accelerated sparse voxel processing)
- **Math / Vectorization**: NumPy, Math
- **Networking**: `websockets`, `asyncio`

### Frontend (User Environment)
- **Framework**: Next.js 14+ (React 18)
- **Language**: TypeScript
- **Styling**: Tailwind CSS, PostCSS
- **WebGL Rendering**: `deck.gl` (by vis.gl)
- **Component UI**: `lucide-react`, standard HTML elements
- **Concurrency Setup**: `concurrently` (for the 1-command startup)

---

## 2. The Data Flow Pipeline (Step-by-Step)

### Step 1: Raw Data Ingestion (Backend)
- The pipeline starts by locating KITTI Odometry Sequence 00.
- `np.fromfile` streams raw `.bin` files into memory as dense matrices of shape `(N, 4)` where columns are `[x, y, z, intensity]`. N averages ~120,000 points per frame.

### Step 2: Voxelisation & Point Extraction
- The points pass into the `CylinderVoxelizer()`. 
- Every cartesian `(x, y, z)` point is mapped to a cylindrical polar coordinate `(rho, phi, z)` relative to the sensor.
- The `grid_ind` function traps these continuous waves into discrete volumetric "cubes" (voxels).
- A 9-dimensional metadata vector `[Δρ, Δφ, Δz, ρ, φ, z, x_cart, y_cart, intensity]` is assigned to every single point.

### Step 3: Cylinder3D SubMConv Neural Inference
- The PyTorch neural network steps in. 
- Using `spconv.SubMConv3d` (Sparse Sub-manifold convolutions), the network evaluates only the blocks where data exists—making the model computationally cheap and blindingly fast natively on CUDA.
- The U-Net encoder-decoder returns predicted *logits* for all 20 SemanticKITTI classes (car, pedestrian, road, tree, fence).
- `argmax(dim=0)` gets the highest probability class per voxel.
- The `REMAP` tensor instantly scales the 20 classes into Kavach's 3 tactical tiers:
  - `0`: Road / Safe
  - `1`: Static Obstacle
  - `2`: Dynamic Target (Active Threat)

### Step 4: 2.5D Cartesian Compression (`build_grid()`)
- It is impossible to send 120,000 points to the browser via WebSocket at speeds >3 FPS smoothly.
- **The Compression:** Instead of treating points individually, we lay a blanket `80m x 40m` 2D grid over the world and divide it into exact 2x2 meter Cartesian cells (like Minecraft chunks).
- Using vectorized `np.unique` mapping, points fall into their relative chunk.
- **Delta-Z Physics:** If the lowest point in a chunk is `-1.7m` (ground) and the highest is `+1.0m`, the object has a total physical height of `2.7m`.
- **Temporal Voting (The 10-Frame Buffer):** To stop the AI engine from flickering class predictions between frames, every cell checks its own "memory buffer". We poll a `collections.deque(maxlen=10)` to take the literal *mode* (most common prediction) over the last 2 seconds. The AI's momentary mistakes are completely erased.

### Step 5: JSON WebSocket Emitter
- The grid physics condense the chunk into exactly 4 `(x, y)` polygon vertices.
- The total response size shrinks from ~2 Megabytes of raw float data down to `~20 Kilobytes` of JSON String data (a **99% reduction**). 
- `asyncio.gather()` blasts the payload string to the Next.js `ws://` client listening at port 8000.

### Step 6: Frontend Parsing & Hydration (`useLidarStream.ts`)
- The Next.js frontend uses a custom React Hook `useLidarStream()` hooked tightly to standard WebSockets.
- Memory saved metrics and hardware latency telemetry are immediately popped into React State and given to the layout components.

### Step 7: WebGL Rendering (DeckGL)
- The JSON array of generic polygon `[ [x0,y0], [x1,y0], ... ]` vertices connects mathematically to `@deck.gl/layers/PolygonLayer`.
- DeckGL hooks directly into the browser's raw OpenGL hardware path to instantly cast thousands of extruded rectangular prisms mapping true physics `Z-height` to UI `elev`.
- The cycle immediately closes and restarts for the next frame.
