import './App.css';
// react router
import { Routes, Route } from "react-router-dom";
import Dashboard from './dashboard';


function App() {
    return (
        <Routes>
            <Route path="/" element={<>
                <>

                    <section className="home container-fluid p-0">
                        <div className="row home-inner m-0">
                            <div className="col-lg-5 side-bar-homepage pb-16 pb-sm-10">
                                <h1 className="font-bolder display-6 p-md-10 px-10 mt-5 mt-lg-auto text-center text-white mb-3">DefenseVision</h1>
                                <div className="content align-content-center justify-content-center mt-lg-14 p-info">
                                    <div className="colss col-12 mb-3 mt-2 h4 text-white">
                                        B.E. Final Year Project - 2023
                                    </div>
                                    {/* project made by information */}
                                    <div className="colss col-12">
                                        Rishabh Rathore - 7444
                                    </div>
                                    <div className="colss col-12">
                                        Ankit Gadhwal - 7423</div>
                                    <div className="colss col-12">
                                        Arvind Jakhar - 3412
                                    </div>
                                    <div className="colss col-12">
                                        Neeraj Singh - 7433
                                    </div>
                                    <div className="colss col-12">
                                        Guided by: Prof. Seeta Yadav
                                    </div>

                                </div>
                            </div>
                            <div className="col-lg-7 d-flex justify-content-around">
                                <div className="p-10 login-box m-lg-20 w-full">
                                    <brand className="">Keeping our troops safe from above<span className="font-bolder text-tertiary">&nbsp;!</span>
                                    </brand>
                                    <h2 className="mt-6">Log into your Account</h2>
                                    <form className="login-form mt-6 w-100 row h-auto">
                                        <div className="mb-3 input-group-md text-xs col-sm-12 col-md-6">
                                            <label className="form-label" htmlFor="email">Email address</label>
                                            <input type="email" className="form-control" id="email" placeholder="Your email address" />
                                        </div>
                                        <div className="mb-3 input-group-md text-xs col-sm-12 col-md-6">
                                            <label className="form-label" >Password</label>
                                            <input type="password" className="form-control" placeholder="Password" />
                                        </div>
                                        <div className="mb-5 text-xs">
                                            <div className="form-check">
                                                <input className="form-check-input" type="checkbox" name="check_example" id="check_example" />
                                                <label className="form-check-label mt-n1 text-xs" htmlFor="check_example">
                                                    Keep me logged in
                                                </label>
                                            </div>
                                        </div>
                                        <div>
                                            <a href="/dashboard" className="btn btn-primary w-full me-5 w-md-max mb-4 my-md-auto">Login</a>
                                        </div>
                                    </form>

                                    <hr className="w-70 mx-auto  mt-10" />

                                </div>
                            </div>
                        </div>
                    </section>

                </>
            </>}></Route>
            <Route path="/dashboard" element={<Dashboard />}></Route>
        </Routes>
    );
}


export default App;
