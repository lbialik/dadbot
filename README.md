# dadbot

## Description

NLP project to generate puns from topic words and sentences! See report.pdf for details on DadBot's implementation.

## Running

```bash
# First, make sure you have Python 3.7 installed. Prefferably install a virtual
# environment, but you do you.

# Install Python requirements
$ pip install -r requirements.txt

# If you're interested in using the web UI, install the Node requirements.
$ cd web-client
$ npm install
```

### CLI

```bash
# Run the word vector server
$ python server.py

# In another window, run the pun generator
$ python main.py

# If you want to use BERT to rerank then
$ python main.py rerank
```

### Web

```bash
# Run the word vector
$ python server.py

# Run the web front-end, and navigate to localhost:3000
$ cd web-client
$ npm start
```

## Contributing

#### Set Up

Please ensure you have the following installed:

* [Python 3.7](https://www.python.org/downloads/) &mdash; This is the version of Python that we'll be using for this project.
    * On Windows, download that link, or the Windows subsystem for Linux. Make sure you add it to your `%PATH%`.
    * On Mac, use [brew](https://brew.sh/).
    * On Linux, you're on your own :)
* [Pre-commit](https://pre-commit.com/) &mdash; We will use this to execute a Python formattter:
    * [black](https://github.com/psf/black) &mdash; Formatting

Follow these steps:

1. Install requirements: `pip install -r requirements.txt`
2. Activate pre-commit: `pre-commit install`

#### Commit Process

When adding code to this repo, please follow the following commit process:

1. Check out your own feature branch: `git checkout -b <branch name>`
2. Commit onto that branch
3. Push to that branch on GitHub: `git push -u origin <branch name>`
4. Ask for a code review from someone else in the group
5. Only merge once someone has approved the code you wrote

This process will help us avoid some of the pitfalls groups run into when they collaborate on complicated projects like this. In particular:

1. It will make sure we maintain high quality code so that it doesn't become cumbersome to work on.
2. It will make sure we are more aware of what each person is working on, so we don't accidentally step on each others' toes.
