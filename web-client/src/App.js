import React from 'react';
import brain from './images/brain.svg';
import './styles/App.css';

const SERVER_HOST = 'localhost';
const SERVER_PORT = 8080;

class App extends React.Component {
  state = {
    topic: '',
    sentence: '',
    output: '',

    loading: false,
  };

  handleText = (name, text) => {
    const d = {};
    d[name] = text;
    this.setState(d);
  };

  punnify = () => {
    const url = `http://${SERVER_HOST}:${SERVER_PORT}/generate_pun`;
    const xhr = new XMLHttpRequest();

    xhr.onload = () => {
      this.setState({
        output: JSON.parse(xhr.response).sentence,
        loading: false,
      });
    };

    xhr.open('POST', url);
    xhr.setRequestHeader('Content-Type', 'application/json');

    this.setState(
      {
        loading: true,
      },
      () => {
        xhr.send(
          JSON.stringify({
            topic: this.state.topic,
            sentence: this.state.sentence,
          }),
        );
      },
    );
  };

  render() {
    return (
      <div className="App">
        <div className="title-section">
          <div className="title">
            <img alt="logo" className="brain" src={brain} />
            DadBot
          </div>
          <div className="subtitle">Computationally Generated Puns</div>
        </div>

        <div className="contents">
          <div>
            <input
              value={this.state.topic}
              onChange={evt => this.handleText('topic', evt.target.value)}
              type="text"
              placeholder="Topic"
            />
          </div>

          <div>
            <input
              value={this.state.sentence}
              onChange={evt => this.handleText('sentence', evt.target.value)}
              type="text"
              placeholder="Sentence"
            />
          </div>

          <div>
            <button onClick={this.punnify} disabled={this.state.loading}>
              Punnify
            </button>
          </div>

          <div>
            <div className="more-like">more like...</div>
            <input
              value={this.state.loading ? 'Loading...' : this.state.output}
              type="text"
              placeholder="Output"
              disabled={true}
            />
          </div>
        </div>
      </div>
    );
  }
}

export default App;
