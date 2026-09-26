import { useMemo } from "react";
import { useGLTF } from "@react-three/drei";
import * as THREE from "three";
import { PageText } from "./reading/PageText";
import { FlippingPage } from "./reading/FlippingPage";
import type { PageSurface } from "./reading/pageSurface";
import type { ReadingChoreography } from "./reading/ReadingChoreography";

export const BOOK_MODEL_URL = "/models/open-book.glb";
const PAGE_MATERIAL_NAME = "Beige";

/** The book model's page material (its pages and cover are separate meshes). */
function findPageMaterial(bookScene: THREE.Object3D): THREE.MeshStandardMaterial {
  const meshes: THREE.Mesh[] = [];
  bookScene.traverse((child) => {
    if (child instanceof THREE.Mesh) meshes.push(child);
  });
  const pageMesh = meshes.find((mesh) => (mesh.material as THREE.Material).name === PAGE_MATERIAL_NAME);
  return (pageMesh?.material as THREE.MeshStandardMaterial | undefined) ?? new THREE.MeshStandardMaterial({ color: "#d9cfb8" });
}

/** "Open Book" by Quaternius (CC0, see public/models/CREDITS.md), with the spread being
 * read printed on its pages, a highlighter under the current word and the page turn. */
export function Book({
  bookScene,
  surface,
  choreography,
}: {
  bookScene: THREE.Object3D;
  surface: PageSurface;
  choreography: ReadingChoreography;
}) {
  const pageMaterial = useMemo(() => findPageMaterial(bookScene), [bookScene]);
  return (
    <>
      <primitive object={bookScene} />
      <PageText surface={surface} choreography={choreography} pageMaterial={pageMaterial} />
      <FlippingPage surface={surface} choreography={choreography} pageMaterial={pageMaterial} />
    </>
  );
}

useGLTF.preload(BOOK_MODEL_URL);
