#!/bin/bash

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================"
echo "LLAMA-2 FINE-TUNING SETUP"
echo "================================"

echo -e "\n${YELLOW}Checking Python version...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.10+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}✗ Python $PYTHON_VERSION found. Required: $REQUIRED_VERSION+${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION detected${NC}"

echo -e "\n${YELLOW}Installing dependencies...${NC}"
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}✗ requirements.txt not found${NC}"
    exit 1
fi

if pip3 install -r requirements.txt --quiet; then
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}✗ Failed to install dependencies${NC}"
    exit 1
fi

echo -e "\n${YELLOW}Checking HuggingFace authentication...${NC}"
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${RED}✗ huggingface-cli not found. Installing...${NC}"
    pip3 install huggingface_hub --quiet
fi

if huggingface-cli whoami &> /dev/null; then
    USER=$(huggingface-cli whoami | head -n1)
    echo -e "${GREEN}✓ Logged in as: $USER${NC}"
else
    echo -e "${YELLOW}⚠ Not logged in to HuggingFace${NC}"
    echo -e "${YELLOW}Please enter your HuggingFace token (get it from https://huggingface.co/settings/tokens):${NC}"
    read -s TOKEN
    echo
    
    if [ -z "$TOKEN" ]; then
        echo -e "${RED}✗ No token provided${NC}"
        exit 1
    fi
    
    if huggingface-cli login --token "$TOKEN"; then
        echo -e "${GREEN}✓ Successfully logged in${NC}"
    else
        echo -e "${RED}✗ Login failed. Please check your token${NC}"
        exit 1
    fi
fi

echo -e "\n${GREEN}================================${NC}"
echo -e "${GREEN}✓ Setup complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo -e "\n${YELLOW}Next step: python prepare_data.py${NC}"
