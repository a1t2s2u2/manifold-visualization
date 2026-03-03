import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export class SceneManager {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls;
  private currentObject: THREE.Object3D | null = null;
  private animationId = 0;
  private clock = new THREE.Clock();
  private fpsCallback: ((fps: number) => void) | null = null;
  private frameCount = 0;
  private fpsTime = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0a0f);

    const sidebarWidth = 320;
    const width = window.innerWidth - sidebarWidth;
    const height = window.innerHeight;
    this.camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
    this.camera.position.set(3, 2, 4);

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
    });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.5;

    this.setupLights();
    this.setupGrid();

    window.addEventListener("resize", () => this.onResize());
    this.animate();
  }

  private setupLights(): void {
    const ambient = new THREE.AmbientLight(0x404060, 0.8);
    this.scene.add(ambient);

    const dir1 = new THREE.DirectionalLight(0x6c5ce7, 1.0);
    dir1.position.set(5, 5, 5);
    this.scene.add(dir1);

    const dir2 = new THREE.DirectionalLight(0xa29bfe, 0.6);
    dir2.position.set(-3, 3, -5);
    this.scene.add(dir2);

    const point = new THREE.PointLight(0xffffff, 0.4);
    point.position.set(0, 5, 0);
    this.scene.add(point);
  }

  private setupGrid(): void {
    const grid = new THREE.GridHelper(10, 20, 0x1a1a2e, 0x1a1a2e);
    grid.position.y = -2;
    (grid.material as THREE.Material).opacity = 0.3;
    (grid.material as THREE.Material).transparent = true;
    this.scene.add(grid);

    const axesHelper = new THREE.AxesHelper(1.5);
    axesHelper.position.set(-3, -2, -3);
    this.scene.add(axesHelper);
  }

  private onResize(): void {
    const sidebarWidth = 320;
    const width = window.innerWidth - sidebarWidth;
    const height = window.innerHeight;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  private animate = (): void => {
    this.animationId = requestAnimationFrame(this.animate);
    const delta = this.clock.getDelta();

    this.controls.update();

    // No per-frame object rotation for point cloud data
    // Auto-rotate is handled by OrbitControls

    this.renderer.render(this.scene, this.camera);

    this.frameCount++;
    this.fpsTime += delta;
    if (this.fpsTime >= 1.0) {
      if (this.fpsCallback) {
        this.fpsCallback(Math.round(this.frameCount / this.fpsTime));
      }
      this.frameCount = 0;
      this.fpsTime = 0;
    }
  };

  setObject(obj: THREE.Object3D): void {
    if (this.currentObject) {
      this.scene.remove(this.currentObject);
      this.disposeObject(this.currentObject);
    }
    this.currentObject = obj;
    this.scene.add(obj);

    const box = new THREE.Box3().setFromObject(obj);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const targetSize = 6;
    const scale = targetSize / maxDim;
    obj.scale.setScalar(scale);
    obj.position.sub(center.multiplyScalar(scale));

    // Auto-fit camera to fill viewport
    const fovRad = (this.camera.fov * Math.PI) / 180;
    const dist = (targetSize / 2) / Math.tan(fovRad / 2) * 1.1;
    this.camera.position.set(dist * 0.55, dist * 0.35, dist * 0.75);
    this.camera.lookAt(0, 0, 0);
    this.controls.target.set(0, 0, 0);
    this.controls.update();

    this.onResize();
  }

  private disposeObject(obj: THREE.Object3D): void {
    obj.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        child.geometry.dispose();
        if (Array.isArray(child.material)) {
          child.material.forEach((m) => m.dispose());
        } else {
          child.material.dispose();
        }
      }
      if (child instanceof THREE.LineSegments || child instanceof THREE.Line) {
        child.geometry.dispose();
        if (Array.isArray(child.material)) {
          child.material.forEach((m) => m.dispose());
        } else {
          child.material.dispose();
        }
      }
      if (child instanceof THREE.Points) {
        child.geometry.dispose();
        if (Array.isArray(child.material)) {
          child.material.forEach((m) => m.dispose());
        } else {
          child.material.dispose();
        }
      }
    });
  }

  getVertexCount(): number {
    let count = 0;
    if (this.currentObject) {
      this.currentObject.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          count += child.geometry.getAttribute("position")?.count ?? 0;
        }
        if (child instanceof THREE.Points) {
          count += child.geometry.getAttribute("position")?.count ?? 0;
        }
      });
    }
    return count;
  }

  onFps(cb: (fps: number) => void): void {
    this.fpsCallback = cb;
  }

  setAutoRotate(enabled: boolean): void {
    this.controls.autoRotate = enabled;
  }

  dispose(): void {
    cancelAnimationFrame(this.animationId);
    this.renderer.dispose();
  }
}
