import React from 'react';
import brain from './images/brain.svg';
import './styles/App.css';

const SERVER_HOST = 'localhost';
const SERVER_PORT = 8080;

const CUTOFF_GOOD_PUN = 1.0;
const CUTOFF_OK_PUN = 1.5;

class App extends React.Component {
  state = {
    topic: '',
    sentence: '',
    output: '',
    cost: -1,

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
      try {
        const json = JSON.parse(xhr.response);
        let avgCost = 0;
        for (let i = 0; i < json.cost.length; i++) {
          avgCost += json.cost[i][1];
        }
        avgCost /= json.cost.length;

        this.setState({
          output: json.sentence,
          cost: avgCost,
          loading: false,
        });
      } catch {
        this.setState({
          output: 'Error!',
          loading: false,
        });
      }
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

  response() {
    if (this.state.cost < 0.0) {
      return '';
    } else if (this.state.cost <= CUTOFF_GOOD_PUN) {
      return '*DadBot giggles*';
    } else if (this.state.cost <= CUTOFF_OK_PUN) {
      return '*DadBot says*';
    } else {
      return '*DadBot hangs his head in shame*';
    }
  }

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

        <form className="contents">
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

            <button onClick={this.punnify} disabled={this.state.loading}>
              Punnify
            </button>
        </form>

        <div className="contents">
          <div>
            <div className="more-like">more like...</div>
            <input
              value={this.state.loading ? 'Loading...' : this.state.output}
              type="text"
              placeholder="Output"
              disabled={true}
            />
          </div>

          <div className="emote">
            {this.response()}
          </div>
        </div>
      </div>
    );
  }
}

export default App;
