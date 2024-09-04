# Function to display usage
usage() {
  echo "Usage: $0 -t HUGGINGFACE_TOKEN -o OPEN_AI_API_KEY" >&2
  exit 1
}

# Parse command-line arguments
while getopts ":t:o:" opt; do
  case $opt in
    t)
      HUGGINGFACE_TOKEN=$OPTARG
      ;;
    o)
      OPENAI_API_KEY=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      usage
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      usage
      ;;
  esac
done

# Check if HUGGINGFACE_TOKEN is set
if [ -z "$HUGGINGFACE_TOKEN" ]; then
  echo "Error: HUGGINGFACE_TOKEN is not set. Please provide it using the -t option."
  usage
fi

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
  echo "Error: OPENAI_API_KEY is not set. Please provide it using the -o option."
  usage
fi

# VSCode extensions
code --install-extension "charliermarsh.ruff" --remote $HOSTNAME
code --install-extension "github.copilot" --remote $HOSTNAME
code --install-extension "github.copilot-chat" --remote $HOSTNAME
code --install-extension "mechatroner.rainbow-csv" --remote $HOSTNAME
code --install-extension "ms-azuretools.vscode-docker" --remote $HOSTNAME
code --install-extension "ms-kubernetes-tools.vscode-kubernetes-tools" --remote $HOSTNAME
code --install-extension "ms-python.debugpy" --remote $HOSTNAME
code --install-extension "ms-python.python" --remote $HOSTNAME
code --install-extension "ms-python.vscode-pylance" --remote $HOSTNAME
code --install-extension "ms-toolsai.jupyter" --remote $HOSTNAME
code --install-extension "ms-toolsai.jupyter-keymap" --remote $HOSTNAME
code --install-extension "ms-toolsai.jupyter-renderers" --remote $HOSTNAME
code --install-extension "ms-toolsai.vscode-jupyter-cell-tags" --remote $HOSTNAME
code --install-extension "ms-toolsai.vscode-jupyter-slideshow" --remote $HOSTNAME
code --install-extension "pkief.material-icon-theme" --remote $HOSTNAME
code --install-extension "redhat.vscode-yaml" --remote $HOSTNAME
code --install-extension "tecosaur.latex-utilities" --remote $HOSTNAME

# Python packages
pip3 install --upgrade pip
pip3 install --upgrade git+https://www.github.com/frank-lab-ai/franky@njf
pip3 install --upgrade accelerate
pip3 install --upgrade datasets
pip3 install --upgrade ipykernel
pip3 install --upgrade kubejobs
pip3 install --upgrade markdown
pip3 install --upgrade networkx
pip3 install --upgrade openai
pip3 install --upgrade pandas
pip3 install --upgrade sentencepiece
pip3 install --upgrade pytest
pip3 install --upgrade tiktoken
pip3 install --upgrade transformers
pip3 install --upgrade torch
pip3 install --upgrade flash-attn

# Git configuration
git config --global user.email "goggled.mapping.0p@icloud.com"
git config --global user.name "njf0"

# Configure Hugging Face CLI with the token
huggingface-cli login --token "$HUGGINGFACE_TOKEN"
export OPENAI_API_KEY="$OPENAI_API_KEY"