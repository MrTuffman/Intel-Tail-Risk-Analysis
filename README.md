# Intel Left-Tail Risk Analysis

## Overview
Estimate Intel’s downside tail index with the Hill estimator and test
whether Gaussian VaR understates extreme-loss frequency.

## Main Findings
- Hill tail index: 2.20
- Selected k: 167
- Gaussian 99% VaR: -6.61%
- Observed breaches: 37
- Expected breaches: 25.1

## Motivation
Explain Intel’s AI, foundry, and data-center turnaround context.

## Methodology
1. Calculate daily log returns
2. Isolate negative returns
3. Estimate the Hill tail index
4. Select k from the stable Hill-plot region
5. Backtest Gaussian 99% VaR

## Repository Structure
Describe each directory.

## Installation
pip install -r requirements.txt

## Usage
python src/hill_intc.py
python scripts/compile_paper.py

## Results
Include the four plots and explain their meaning.

## Limitations
Discuss threshold sensitivity, non-stationarity, and volatility clustering.

## Disclaimer
This project is educational and is not investment advice.
