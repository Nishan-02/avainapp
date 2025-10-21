// src/components/Team.js
import React from 'react';

// Component for an individual team member card
function TeamMemberCard({ name, photoUrl, linkedIn, github }) {
  return (
    <div className="team-member-card">
      <img src={photoUrl} alt={`Profile of ${name}`} />
      <h3>{name}</h3>
      <div className="links">
        {linkedIn && <a href={linkedIn} target="_blank" rel="noopener noreferrer"><i className="fab fa-linkedin"></i></a>}
        {github && <a href={github} target="_blank" rel="noopener noreferrer"><i className="fab fa-github"></i></a>}
      </div>
    </div>
  );
}

// Main Team component
function Team() {
  // ** IMPORTANT **
  // 1. Add your team member photos to the `public/images/team/` folder.
  // 2. Update the array below with your team's info.
  const teamData = [
    { 
      name: "Nishan", 
      photoUrl: "/images/nishan.jpeg", // Path relative to the 'public' folder
      linkedIn: "https://linkedin.com/in/nishann-s", 
      github: "https://github.com/Nishan-02" 
    },
    { 
      name: "Shishir", 
      photoUrl: "/images/shishir.jpeg", // Use .jpg, .png, etc.
      linkedIn: "https://www.linkedin.com/in/shishir-r-kulal-4757a9296?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app", 
      github: "https://github.com/shishir-sh26" 
    },
    { 
      name: "Shrinidhi", 
      photoUrl: "/images/shrinidhi.jpeg",
      linkedIn: "https://www.linkedin.com/in/shrinidhi-anchan", 
      github: "https://github.com/shrinidhianchan"
    },
    { 
      name: "Jaswin", 
      photoUrl: "/images/jaswin.png",
      linkedIn: "https://www.linkedin.com/in/jaswin-c-a-82493632b?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app ",
      github: "https://github.com/Jaswinn"
    }
  ];

 return (
    <section id="team" className="section">
      <h2>Meet The team</h2> {/* Title changed */}
      <div className="team-members-grid">
        {teamData.map(member => (
          <TeamMemberCard 
            key={member.name}
            {...member} // Shorthand for passing all props
          />
        ))}
      </div>
    </section>
  );
}

export default Team;