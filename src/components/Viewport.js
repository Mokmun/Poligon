'use client'
import { Canvas, useLoader } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader";
import { useEffect, useState } from "react";


const Viewport = ({ objPath }) => {
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    console.log("Object Path received in Viewport:", objPath);
    setLoading(false);
  }, [objPath]);

  const Model = () => {
    const obj = useLoader(OBJLoader, objPath);
    return <primitive object={obj} scale={1} />;
  };
  
  const handleDownload = () => {
    const link = document.createElement("a");
    link.href = objPath; // Ensure the file is in the public directory
    link.download = objPath.split('/').pop();
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }
  return (
    <div className="w-3/5 border border-solid border-gray-600 rounded-xl h-4/5 bg-gray-900 flex justify-center items-evenly relative z-10">
      {loading && objPath ? 
      <div className="mt-4 flex items-center space-x-2">
          <div className="animate-spin h-6 w-6 border-t-2 border-blue-500 rounded-full"></div>
          <span className="text-gray-100">Processing file...</span>
      </div> : 
        <Canvas className="w-full h-full">
          <ambientLight intensity={0.5} />
          <directionalLight position={[5, 5, 5]} intensity={1} />
          {objPath && <Model />} 
          <OrbitControls />
        </Canvas>
      
      }
      <button className="absolute bottom-0 right-0 m-4 p-2 bg-violet-500 text-white rounded-md" onClick={handleDownload}>Download</button>
    </div>
  );
};

export default Viewport;