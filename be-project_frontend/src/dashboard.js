import React from "react";
import { useState } from "react";
import { useEffect } from "react";
import arch1 from "./assets/arch1.jpeg";
import arch2 from "./assets/12.png";
import arch3 from "./assets/modulesAndStem.png";
import scaling from "./assets/scaling.png";


const data = [
    {
        id: 1,
        imageAddress: "https://images.unsplash.com/photo-1519074031893-210d39bd3885?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2524&q=80",
        timestamp: "2021-06-01 12:00:00",
        location: "28.7041° N, 77.1025° E",
        result: "No Threats Detected"
    },
    {
        id: 2,
        imageAddress: "https://images.unsplash.com/photo-1629032563246-19ee3aeb7dc4?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2532&q=80",
        timestamp: "2021-06-01 12:00:00",
        location: "28.7041° N, 77.1025° E",
        result: "Planes & Helicopters Detected"
    },
    {
        id: 3,
        imageAddress: "https://images.unsplash.com/photo-1620059116993-398c21ce8406?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2370&q=80",
        timestamp: "2021-06-01 12:00:00",
        location: "28.7041° N, 77.1025° E",
        result: "Bunkers Detected"
    },
    // generate 10 more with random result
    {
        id: 4,
        imageAddress: "https://plus.unsplash.com/premium_photo-1661875250660-5cb2e21e5a56?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2370&q=80",
        timestamp: "2021-06-01 12:00:00",
        location: "28.7041° N, 77.1025° E",
        result: "No Threats Detected"
    },
    {
        id: 5,
        imageAddress: "https://images.unsplash.com/photo-1535332489047-1170f3fff966?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2369&q=80",
        timestamp: "2021-06-01 12:00:00",
        location: "28.7041° N, 77.1025° E",
        result: "Planes & Helicopters Detected"
    },
    {
        id: 6,
        imageAddress: "https://images.unsplash.com/photo-1635794675394-8246d343234b?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MTV8fG1pbGl0YXJ5JTIwYnVua2VyfGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=800&q=60",
        timestamp: "2021-06-01 12:00:00",
        location: "28.7041° N, 77.1025° E",
        result: "Bunkers Detected"
    },
    {
        id: 7,
        imageAddress: "https://images.unsplash.com/photo-1629032642277-a112ec9fe93b?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=987&q=80",
        timestamp: "2021-06-01 12:00:00",
        location: "28.7041° N, 77.1025° E",
        result: "No Threats Detected"
    }
]

function Dashboard() {
    const [imageData, setImageData] = useState([]);
    const [loading, setLoading] = useState(false);
    const [predictions, setPredictions] = useState();

    const onImageChange = (e) => {
        setImageData([e.target.files]);
        console.log(imageData);
    }

    function SendImage(){
        console.log(imageData[0][0]);
        setLoading(true);
        // send image to backend
        var formdata = new FormData();
        formdata.append("image", imageData[0][0], "i1.jpeg");

        var requestOptions = {
            method: 'POST',
            body: formdata,
            redirect: 'follow'
        };

        fetch("http://127.0.0.1:5000/predict", requestOptions)
            .then(response => response.json())
            .then(result => {
                console.log(result.prediction)
                setLoading(false)
                setPredictions(result.prediction)
                })
            .catch(error => {
                console.log('error', error)
                setLoading(false)
                setPredictions("No Military Objects Detected" )
            });

    }
    
    return (
        <>
                {/* Navbar */}
                <nav className="navbar navbar-expand-lg navbar-light bg-light px-md-2 px-0">
                    <div className="container-fluid m-0 px-3">
                        <button className=" font-bolder display-7 text-center">DefenseVision</button>

                        <div className="d-flex">
                            <button className="btn btn-sm btn-outline-primary me-2">Scan New Image</button>
                        </div>
                    </div>
                </nav>
          
            {/* Already Scaned Images */}
            {/* <div className="container-fluid mt-5 px-md-5 px-2">
                <p className="small-heading px-5">Scanned History</p>
                <div className="row m-0 mt-4 justify-content-start align-content-start align-itmes-start">
                    { // map through all the images
                        data.map((item) => {
                            return (
                                <div className="col-12 col-md-6 col-lg-4 col-xl-3 scanned-img mb-10" key={item.id}>
                                    <img src={item.imageAddress} alt="img" className="img-fluid" />
                                    <div className="img-info">
                                        <p className="small-heading">
                                            {item.result}
                                        </p>
                                        <p className="smaller-heading">{item.location}, {item.timestamp}</p>
 
                                    </div>
                                </div>
                            )
                        }
                        )
                    }
                </div>
            </div> */}

            <div className={imageData.length > 0 ? "container-fluid mt-8 px-md-5 px-2 scanContainer" : "container-fluid mt-8 px-md-5 px-2 scanContainer noImage" }>
                <p className="small-heading px-5">Scan New Image</p>
                {/* select image */}
                {imageData.length > 0 ?
                <>
                    <img src={URL.createObjectURL(imageData[0][0])} className="selectedImage" alt="wow"/> 
                    {predictions ? <p>{predictions}</p> : <></>}
                    {}
                </>
                    
                    :
                    <div>
                        <input type="file" id="file" accept="image/*" onChange={onImageChange} className="inputfile"/>
                        <label htmlFor="file" className="btn btn-sm btn-outline-primary me-2">Select Image</label>
                    </div>
                }
                {imageData.length > 0 &&
                ( loading ? 
                <button className="btn btn-sm btn-outline-primary me-2"
                onClick={SendImage}
                >Scanning...</button>

                :
                <>
                <button className="btn btn-sm btn-outline-primary me-2"
                onClick={SendImage}
                >Scan</button>
                <button className="btn btn-sm btn-outline-primary me-2"
                onClick={() => {
                    setImageData([])
                    setPredictions()

                }}
                >Change Image</button>
                </>
                )
            }
            </div>

            <div className="container-fluid mt-10 px-md-5 px-2 scanContainer2">
                <h4 className="mb-6">
                    How DefenseVision Works?
                </h4>
                <img src={arch1} alt="arch1" className="img-fluid" />
                <hr />
                <h4 className="mb-6">
                    Scaling in EffcientNet
                </h4>
                <img src={scaling} alt="arch4" className="img-fluid" />
                <hr />
                <h4 className="mb-6">
                    EffcientNet Architecture
                </h4>
                <img src={arch2} alt="arch2" className="img-fluid" />
                <hr />
                <img src={arch3} alt="arch3" className="img-fluid" />
            </div>

            <div className="container-fluid mt-10 px-md-5 px-2 scanContainer3">
                <h4 className="mb-6">
                    Project Team
                </h4>
                <ul>
                    <li>Rishabh Rathore</li>
                    <li>Ankit Gadhwal</li>
                    <li>Neerak Singh</li>
                    <li>Arvind Kumar</li>
                </ul>
                <h4 className="mb-6">
                    Project Mentors : 
                </h4>
                <ul>
                    <li>Prof. Seeta Yadav</li>
                    <li>Prof. Anup Kadam</li>
                    <li>Dr. Sunil Dhore</li>
                </ul>

            </div>


        </>
    );
}

export default Dashboard;
