#!/bin/bash

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "================================"
echo "LLAMA-2 FINE-TUNING PIPELINE"
echo "================================"

echo -e "\n${YELLOW}[1/3] Setup${NC}"
if [ ! -f ".setup_complete" ]; then
    bash setup.sh || {
        echo -e "${RED}✗ Setup failed${NC}"
        exit 1
    }
    touch .setup_complete
else
    echo -e "${GREEN}✓ Setup already complete (skip)${NC}"
fi
echo -e "${GREEN}✓ Setup complete${NC}"

echo -e "\n${YELLOW}[2/3] Prepare data${NC}"
if [ ! -d "processed_data" ]; then
    python3 prepare_data.py || {
        echo -e "${RED}✗ Data preparation failed${NC}"
        exit 1
    }
else
    echo -e "${GREEN}✓ Data already prepared (skip)${NC}"
fi
echo -e "${GREEN}✓ Data ready${NC}"

echo -e "\n${YELLOW}[3/3] Train model (6-8 hours)${NC}"
echo -e "${YELLOW}⚠ This will use significant GPU/CPU resources${NC}"
read -p "Start training? (y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 train_llama2.py || {
        echo -e "${RED}✗ Training failed${NC}"
        exit 1
    }
else
    echo ""
    echo -e "${YELLOW}Training cancelled.${NC}"
    echo -e "${YELLOW}Run 'python3 train_llama2.py' when ready.${NC}"
    exit 0
fi

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}✓ TRAINING COMPLETE!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo -e "Model saved to: ${YELLOW}./red_team_llama2/${NC}"
echo ""
echo "Next steps:"
echo "  1. Test the model with inference scripts"
echo "  2. Evaluate on held-out test sets"
echo "  3. Deploy for red-teaming applications"
echo ""
