"use client";
import Image from "next/image";
import FileUpload from "@/components/FileUpload";
import Viewport from "@/components/Viewport";
import Loading from "@/components/Loading";
import { useState } from "react";
import { useAuth } from "@clerk/nextjs";
import Link from "next/link";
const Home = () => {
  const [objPath, setObjPath] = useState("");
  const [loading, setLoading] = useState(false);

  return (
    <>
      {loading ? <Loading /> : null}
      <div className="relative flex flex-col lg:flex-row w-full items-center justify-center h-full lg:h-screen pb-16 mt-6">
        <div className="flex w-full lg:w-1/3 items-center justify-center h-full px-6">
          <div className="flex w-full h-full flex-col items-center justify-center">
            <FileUpload setObjPath={setObjPath} setLoading={setLoading} />
          </div>
        </div>
        <Viewport objPath={objPath} />
        {/* // add loading components */}
      </div>
    </>
  );
};

export default Home;
