"use client";
import axios from "axios";
import { useEffect, useState } from "react";
import { useAuth } from "@clerk/nextjs";
import Card from "@/components/Card";

const page = () => {
  const [collections, setCollections] = useState([]);
  const [loading, setLoading] = useState(true);
  const { userId } = useAuth();

  useEffect(() => {
    const fetchCollection = async () => {
      try {
        console.log(userId);
        const response = await axios.get(
          `https://poligon-mkk.vercel.app/collection/${userId}`
        );
        console.log(response.data);
        setCollections(response.data);
      } catch (error) {
        console.error("Fetch collection error:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchCollection();
  }, [userId]);

  const handleDelete = (fileName) => {
    setCollections((prevCollections) =>
      prevCollections.filter((item) => item.file_name !== fileName)
    );
  };

  if (loading)
    return <p className="text-center text-gray-500">Loading collections...</p>;
  if (!collections.length)
    return <p className="text-center text-gray-500">No collections found.</p>;

  return (
    <div className="relative flex w-full h-full justify-center items-center flex-col container m-auto pb-24 mt-12">
      <h1 className="text-3xl text-left font-bold mb-4">Collection</h1>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 justify-center">
        {collections.map((item, index) => (
          <div
            key={index}
            className="border rounded-lg p-4 shadow-lg m-4 gap-4"
          >
            <Card
              key={index}
              userId={userId}
              imageUrl={item.imgfile_url}
              fileName={item.file_name}
              objUrl={item.objfile_url}
              onDelete={handleDelete}
            />
          </div>
        ))}
      </div>
    </div>
  );
};

export default page;
