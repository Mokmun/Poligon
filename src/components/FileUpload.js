"use client";
import React, { useState } from "react";
import axios from "axios";
import { useAuth } from "@clerk/nextjs";
const FileUpload = ({ setObjPath, setLoading }) => {
    const { userId } = useAuth(); 
    const [file, setFile] = useState(null);
    const [uploadResponse, setUploadResponse] = useState(null);

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        if (!file) return alert("Please select a file first.");

        setLoading(true); // Set loading to true when upload starts

        const formData = new FormData();
        formData.append("file", file);
        formData.append("filename", document.getElementById("filename").value);
        formData.append("user_id", userId);
        formData.append("mod3l", document.getElementById("model").value);
        formData.append("category", document.getElementById("category").value);
        try {
            const response = await axios.post("http://localhost:8000/predict/", formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });

            console.log("API Response:", response.data);

            if (response.status === 200 && response.data.filename) {
                setUploadResponse(`File uploaded: ${response.data.filename}`);
                setObjPath(`${response.data.filename}`);
            } else {
                setUploadResponse("Upload failed");
            }
        } catch (error) {
            console.error("Upload error:", error);
            setUploadResponse("Upload failed. Check console.");
        } finally {
            setLoading(false); // Set loading to false when response is received
        }
    };

    return (
        <div className="flex flex-col items-center justify-center mb-12 ">
            <form onSubmit={handleSubmit} className="bg-gray-800 p-6 rounded-lg shadow-md border-solid border-gray-600 border">
                <div className="mb-4">
                    <label className="block text-pink-500 text-lg font-bold mb-2" htmlFor="filename">
                        Filename
                    </label>
                    <input
                        type="text"
                        id="filename"
                        className="w-full px-3 py-2 border rounded-lg text-black focus:outline-none focus:border-pink-500 mb-4"
                        required
                    />    
                    <label className="block text-pink-500 text-lg font-bold mb-2" htmlFor="category">
                        Select Model
                    </label>
                    <select
                        id="model"
                        className="w-full px-3 py-2 border rounded-lg text-black focus:outline-none focus:border-pink-500 mb-4"
                    >
                        <option value="NMR">NMR</option>
                        <option value="POCO">NMR + POCO</option>
                    </select>
                    <label className="block text-pink-500 text-lg font-bold mb-2" htmlFor="category">
                        Select category
                    </label>
                    <select
                        id="category"
                        className="w-full px-3 py-2 border rounded-lg text-black focus:outline-none focus:border-pink-500 mb-4"
                    >
                        <option value="airplane">Airplane</option>
                        <option value="bench">Bench</option>
                        <option value="dresser">Dresser</option>
                        <option value="car">Car</option>
                        <option value="chair">Chair</option>
                        <option value="display">Display</option>
                        <option value="lamp">Lamp</option>
                        <option value="speaker">Speaker</option>
                        <option value="rifle">Rifle</option>
                        <option value="sofa">Sofa</option>
                        <option value="table">Table</option>
                        <option value="phone">Phone</option>
                        <option value="vessel">Bench</option>
                    </select>
                    <label className="block text-pink-500 text-lg font-bold mb-2" htmlFor="file">
                        Upload File
                    </label>
                    <input
                        type="file"
                        id="file"
                        onChange={handleFileChange}
                        className="w-full px-3 py-2 border rounded-lg text-white focus:outline-none focus:border-pink-500"
                    />
                </div>

                <button
                    type="submit"
                    className={`w-full py-2 px-4 rounded-lg btn-primary`}
                >
                    Upload
                </button>
            </form>

            {/* Show response message */}
            {uploadResponse && <p className="mt-4 text-white">{uploadResponse}</p>}
        </div>
    );
};

export default FileUpload;
