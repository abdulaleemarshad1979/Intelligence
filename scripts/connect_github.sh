#!/usr/bin/env bash
set -e

# ==============================================================================
# CCTV Intelligence — GitHub Account Connection & Git Initialization Utility
# User: abdulaleemarshad1979 <abdulaleemarsahdm@gmail.com>
# ==============================================================================

echo "================================================================================"
echo " CCTV INTELLIGENCE — CONNECT GITHUB ACCOUNT"
echo "================================================================================"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

GIT_USERNAME="abdulaleemarshad1979"
GIT_EMAIL="abdulaleemarsahdm@gmail.com"
GITHUB_REPO_SSH="git@github.com:${GIT_USERNAME}/cctv-intelligence.git"
GITHUB_REPO_HTTPS="https://github.com/${GIT_USERNAME}/cctv-intelligence.git"

# 1. Configure Global Git Identity
echo "[1/5] Configuring Git identity..."
git config --global user.name "$GIT_USERNAME"
git config --global user.email "$GIT_EMAIL"
git config --global init.defaultBranch main
git config --global pull.rebase false
echo "      User Name:  $(git config --global user.name)"
echo "      User Email: $(git config --global user.email)"

# 2. Initialize Local Git Repository if not present
echo "[2/5] Initializing Git repository in $REPO_DIR..."
if [ ! -d ".git" ]; then
    git init -b main
    echo "      Initialized fresh Git repository (branch: main)."
else
    echo "      Git repository already initialized."
fi

# 3. Setup SSH Key for secure, passwordless GitHub connection
echo "[3/5] Checking SSH keys..."
SSH_DIR="$HOME/.ssh"
KEY_FILE="$SSH_DIR/id_ed25519"
mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"

if [ ! -f "$KEY_FILE" ]; then
    echo "      Generating new ED25519 SSH key for $GIT_EMAIL..."
    ssh-keygen -t ed25519 -C "$GIT_EMAIL" -f "$KEY_FILE" -N ""
    chmod 600 "$KEY_FILE"
    chmod 644 "${KEY_FILE}.pub"
    echo "      SSH key generated successfully."
else
    echo "      Existing SSH key found at $KEY_FILE."
fi

# 4. Set Remote Origin
echo "[4/5] Configuring remote origin..."
if git remote | grep -q "origin"; then
    CURRENT_REMOTE=$(git remote get-url origin)
    echo "      Current remote origin: $CURRENT_REMOTE"
else
    git remote add origin "$GITHUB_REPO_SSH"
    echo "      Added remote origin: $GITHUB_REPO_SSH"
fi

# 5. Display Connection Instructions & Public Key
echo "[5/5] GitHub Account Link Instructions:"
echo "================================================================================"
echo " YOUR PUBLIC SSH KEY (Copy the line below):"
echo "--------------------------------------------------------------------------------"
cat "${KEY_FILE}.pub"
echo "--------------------------------------------------------------------------------"
echo ""
echo " STEPS TO LINK YOUR GITHUB ACCOUNT:"
echo " 1. Open your browser and go to: https://github.com/settings/keys"
echo " 2. Click 'New SSH Key'"
echo " 3. Title: 'Workstation Ubuntu CCTV'"
echo " 4. Paste the key shown above and click 'Add SSH key'"
echo ""
echo " 5. Create the repository on GitHub if you haven't already:"
echo "    https://github.com/new -> Name: cctv-intelligence (Public or Private)"
echo ""
echo " 6. Test authentication with:"
echo "    ssh -T git@github.com"
echo ""
echo " 7. Push your project:"
echo "    git add ."
echo "    git commit -m 'feat: infuse neural models from GitHub & investigation graph'"
echo "    git push -u origin main"
echo "================================================================================"
