import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

// Expected model frame (see README): metres, +Y up, toe toward +Z, origin at
// the bottom of the heel, modelled as a right shoe.
export const MODEL_URL = `${import.meta.env.BASE_URL}models/shoe.glb`;

// Stand-in shoe until the 3DEXPERIENCE export is dropped into public/models.
function placeholderShoe() {
  const group = new THREE.Group();
  const length = 0.28;
  const soleMat = new THREE.MeshStandardMaterial({ color: 0xf2f2f2, roughness: 0.8 });
  const upperMat = new THREE.MeshStandardMaterial({ color: 0x2b6cb0, roughness: 0.5 });

  const sole = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.025, length), soleMat);
  sole.position.set(0, 0.0125, length / 2);
  group.add(sole);

  const upper = new THREE.Mesh(new THREE.CapsuleGeometry(0.045, length - 0.1, 8, 16), upperMat);
  upper.rotation.x = Math.PI / 2;
  upper.scale.set(1, 1, 0.9);
  upper.position.set(0, 0.06, length / 2);
  group.add(upper);

  const collar = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.05, 0.07, 24, 1, true), upperMat);
  collar.position.set(0, 0.1, 0.06);
  group.add(collar);
  return group;
}

// Resolves to { object, length, isPlaceholder } where length is the model's heel-to-toe
// extent in metres.
export async function loadShoe() {
  let object;
  let isPlaceholder = false;
  try {
    const gltf = await new GLTFLoader().loadAsync(MODEL_URL);
    object = gltf.scene;
  } catch {
    object = placeholderShoe();
    isPlaceholder = true;
  }
  const box = new THREE.Box3().setFromObject(object);
  const length = box.max.z - box.min.z || 0.28;
  return { object, length, isPlaceholder };
}
