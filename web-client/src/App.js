import React from 'react';
import brain from './images/brain.svg';
import './styles/App.css';

const SERVER_HOST = 'localhost';
const SERVER_PORT = 3000;

function App() {
  return (
    <div className="App">
      <div className="title-section">
        <div className="title">
          <img className="brain" src={brain} />
          DadBot
        </div>
        <div className="subtitle">Computationally Generated Puns</div>
      </div>

      <div className="contents">
        <div>
          <input type="text" placeholder="Topic" />
        </div>

        <div>
          <input type="text" placeholder="Sentence" />
        </div>

        <div>
          <button>Punnify</button>
        </div>

        <div>
          <div className="more-like">more like...</div>
          <input type="text" placeholder="Output" disabled="true" />
        </div>
      </div>
    </div>
  );
}

export default App;
