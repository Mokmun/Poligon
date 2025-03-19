from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import List
import chainer
import chainer.serializers
import numpy as np
import skimage.io
import neural_renderer
import io
import open3d as o3d
import os
from fastapi.middleware.cors import CORSMiddleware
from models import Model
from schema import UserSchema
from bson.objectid import ObjectId
from database import user, collection
from util import upload_file_to_s3, delete_file_from_s3
from io import BytesIO
from mesh_recon import mesh_recon

app = FastAPI()

class UserSchema(BaseModel):
    id: str
    username: str
    email: str
    collection: list

class CollectionSchema(BaseModel):
    user_id: str
    file_name: str
    imgfile_url: str
    objfile_url: str



@app.post("/api/webhook/clerk")
async def clerk_webhook(request: Request):
    """Receive Clerk webhook and store user data"""
    try:
        payload = await request.json()

        user_data = payload.get("data", {})
        clerk_id = user_data.get("id")
        email = user_data.get("email_addresses", [{}])[0].get("email_address", "unknown")
        username = user_data.get("username", "unknown")
        event = payload.get("type", "")

        if not clerk_id:
            raise HTTPException(status_code=400, detail="Missing Clerk ID")
        
        existing_user = await user.find_one({"id": clerk_id})
        if existing_user and event == "user.created":
            return {"message": "User already exists"}

        if event == "user.created":
            new_user = {"id": clerk_id, "email": email, "username": username, "collection": []}
            await user.insert_one(new_user)
        elif existing_user and event == "user.updated":
            await user.update_one({"id": clerk_id}, {"$set": {"email": email, "username": username}})
        elif event == "user.deleted":
            await user.delete_one({"id": clerk_id})
            return {"message": "User deleted successfully"}
        else:
            return {"message": "Invalid event type"}

        return {"message": "User added successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Allow frontend requests from Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", 'https://dassie-pumped-vertically.ngrok-free.app'],  # Next.js default port, ngrox domain for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


PUBLIC_DIR = "../public"

@app.post("/predict/")
async def predict(file: UploadFile = File(...), filename: str = Form(...), user_id: str = Form(...), mod3l: str = Form(...), category: str = Form(...)):

    # Load Chainer model
    model = Model()
    model.to_gpu()
    chainer.serializers.load_npz(f"models/{category}.npz", model)

    # Read uploaded image
    image_bytes = await file.read()
    image = skimage.io.imread(io.BytesIO(image_bytes)).astype('float32') / 255
    image = skimage.transform.resize(image, (64, 64), anti_aliasing=True)

    # Ensure image is RGBA
    if image.ndim != 3:
        return {"error": "Input must be an RGBA image."}

    images_in = image.transpose((2, 0, 1))[None, :, :, :]
    images_in = chainer.cuda.to_gpu(images_in)

    # Define file names
    base_filename = filename
    obj_filename = f"{base_filename}.obj"
    obj_model_path = os.path.join(PUBLIC_DIR, obj_filename)


    # Predict 3D model
    if mod3l == "NMR":
        vertices, faces = model.reconstruct(images_in)
        neural_renderer.save_obj(obj_model_path, vertices.data.get()[0], faces.get()[0])
    else:
        print(mod3l)
        vertice = model.reconstruct_pc(images_in)
        mesh = mesh_recon(vertice)

        if isinstance(mesh, o3d.geometry.TriangleMesh):
            vertices = np.asarray(mesh.vertices)  # Convert vertices to NumPy
            faces = np.asarray(mesh.triangles)   # Convert faces to NumPy


        print(f"Vertices shape: {vertices.shape} - Type: {type(vertices)}")
        print(f"Faces shape: {faces.shape} - Type: {type(faces)}")
        neural_renderer.save_obj(obj_model_path, vertices, faces)
        

    # Upload the files to S3
    img_file = BytesIO(image_bytes)
    img_path = upload_file_to_s3(img_file, f"{base_filename}.png")

    # Open and upload the .obj file to S3
    with open(obj_model_path, "rb") as obj_file:
        obj_path = upload_file_to_s3(obj_file, obj_filename)

    # Store metadata in database
    metadata = {
        "user_id": user_id,
        "file_name": f"{base_filename}.png",
        "imgfile_url": img_path,
        "objfile_url": obj_path
    }
    
    await collection.insert_one(metadata)
    await user.update_one(
        {"id": user_id},
        {"$push": {"collection": metadata}}
    )

    print("user_id", user_id)

    return {"img_url": img_path, "obj_url": obj_path, "user_id": user_id, "filename": obj_filename}


@app.get("/collection/{user_id}")
async def get_collection(user_id: str , response_model=List[CollectionSchema]):
    """Retrieve user's collection"""
    user_data = await user.find_one({"id": user_id})
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    data = user_data.get("collection", [])
    formatted_collection = [
    CollectionSchema(**item) for item in data
    ]
    print("formatted_collection", formatted_collection)
    return formatted_collection

@app.delete("/collection/{user_id}/{file_name}")
async def delete_model(user_id: str, file_name: str):
    """Delete a model from user's collection"""
    user_data = await user.find_one({"id": user_id})
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    collection_data = user_data.get("collection", [])
    for item in collection_data:
        if item.get("file_name") == file_name:
            await collection.delete_one({"file_name": file_name})
            await user.update_one(
                {"id": user_id},
                {"$pull": {"collection": {"file_name": file_name}}}
            )
    return delete_file_from_s3(file_name.split(".")[0])
    

    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
## runserver: uvicorn backend.main:app --reload --host localhost --port 8000
## http://localhost:8000/
## http://localhost:8000/docs
