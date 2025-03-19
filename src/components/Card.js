import PropTypes from "prop-types";
import axios from "axios";
const Card = ({ userId, imageUrl, fileName, objUrl, onDelete }) => {
    const deleteCollection = async () => {
        try {
            const response = await axios.delete(`http://localhost:8000/collection/${userId}/${fileName}`)
            console.log(response.data)
            onDelete(fileName);
        }
        catch (error) {
            console.error("Delete collection error:", error);
        }
    }
    return (
        <div className="border rounded-lg shadow-lg overflow-hidden p-4 bg-white">
            <img 
                src={imageUrl} 
                alt={fileName} 
                className="w-full h-40 object-cover rounded-md mb-4"
            />
            <h2 className="text-black text-lg font-semibold mt-2">{fileName}</h2>
            <div className="flex justify-end items-center w-full gap-4 px-2">
                <button className="bg-blue-500 hover:bg-blue-700 px-4 py-2 rounded-lg">
                    <a 
                        href={objUrl} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="text-white hover:underline block"
                    >
                        Download
                    </a>
                </button>
                <button className="bg-red-500 hover:bg-red-700 px-4 py-2 rounded-lg" onClick={deleteCollection}>
                    Delete
                </button>
            </div>
        </div>
    );
}

// Define PropTypes
Card.propTypes = {
    imageUrl: PropTypes.string.isRequired,
    fileName: PropTypes.string.isRequired,
    objUrl: PropTypes.string.isRequired,
    onDelete: PropTypes.func.isRequired,
};


export default Card;