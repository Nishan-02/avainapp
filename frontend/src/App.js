// src/App.js
import React, { useState, useEffect } from 'react';
import './App.css'; 

import Header from './components/Header';
import About from './components/About';
import Predictor from './components/Predictor';
import Team from './components/Team';
import Footer from './components/Footer';

function App() {
  const [isAppLoading, setIsAppLoading] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsAppLoading(false);
    }, 1500); // 1.5-second loading screen

    return () => clearTimeout(timer);
  }, []);

  if (isAppLoading) {
    return (
      <div className="page-loader">
        <div className="loader-logo">
          <i className="fas fa-broadcast-tower"></i>
        </div>
        <div className="loader-spinner"></div>
        <h1>Avian Weather Net</h1>
      </div>
    );
  }

 // ... inside the return()
  return (
    <div className="App">
      <Header />
      <main>
        <About />
        <Predictor />
        <Team /> {/* <-- ADD THIS LINE */}
      </main>
      <Footer />
    </div>
  );
// ...
}

export default App;