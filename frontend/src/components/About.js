// src/components/About.js
import React from 'react';

function About() {
  return (
    <section id="about" className="section">
      <h2>About</h2>
      <p style={{ textAlign: 'justify', fontSize: '1.05rem', lineHeight: '1.7' }}>
        Avian Weather Net is an innovative project that leverages the power of deep learning to predict local weather conditions based on the sounds of birds. Our system analyzes audio recordings of birdsong, using a suite of six specialized deep learning models to classify the weather into one of five categories: Sunny, Rainy, Cloudy, Windy, or Foggy.
      </p>
    </section>
  );
}

export default About;