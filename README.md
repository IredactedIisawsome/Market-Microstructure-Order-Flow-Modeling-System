# Market-Microstructure-Order-Flow-Modeling-System

# **Overview**

This project is a market microstructure research and engineering system focused on processing high frequency trade and Level 2 order book data to study short horizon order flow dynamics. The system emphasizes data quality, non stationarity, and realistic evaluation rather than point prediction alone.

The goal of this repository is to demonstrate system design, modeling philosophy, and evaluation techniques used when working with noisy, adversarial data. Strategy specific logic and parameters are intentionally omitted.

The full implementation is maintained in a private repository and is available to recruiters upon request.

# **System Architecture**

The system is structured as an end to end pipeline:

High frequency trade and Level 2 order book ingestion

Filtration layer to remove microstructure noise and low information events

Feature extraction from depth and order flow dynamics

Regime segmentation to handle non stationarity

Distributional modeling using quantile based approaches

Backtesting and evaluation under execution constraints

Each stage is designed to be modular, testable, and configurable.

# **Data Sources**

The system operates on:

Trade prints

Level 2 bid and ask depth across multiple price levels

Level 2 depth is used to model liquidity distribution, depth imbalance, and potential execution impact rather than relying only on top of book information.

No real market data is included in this public repository.

Filtration Layer

Raw market data contains significant structural noise. A dedicated filtration layer is applied before any feature generation or modeling.

The filtration process removes or downweights:

Stale quotes

Crossed or locked markets

Mechanically induced bursts of low information activity

Events unlikely to contribute to meaningful short horizon signal

This layer is critical for improving signal to noise ratio and ensuring downstream models are trained on consistent inputs.

Exact filtration rules are not disclosed in this public version.

##**Level 2 Order Book Modeling**

Rather than treating the order book as a single price and size, the system models liquidity across multiple depth levels.

This enables:

Measurement of depth shape and imbalance

Estimation of sweep cost and liquidity consumption

Analysis of how depth distribution influences short horizon price behavior

All depth based features are abstracted in this repository.

# **Regime Segmentation**

Market behavior is non stationary. To address this, the system segments data into regimes based on market conditions such as liquidity and volatility.

Models and evaluation metrics are analyzed both globally and within regimes to avoid overestimating performance in favorable conditions.

Specific regime definitions and boundaries are omitted.

# **Distributional Modeling**

Instead of predicting a single outcome, the system uses quantile based modeling to estimate conditional return distributions.

This approach supports:

Asymmetric decision making

Risk aware thresholding

More informative evaluation than point forecasts

Quantile regression is used as a core modeling tool, though model internals are not included here.

# **Evaluation and Backtesting**

Model performance is evaluated using:

Macro F1 for classification balance

Expected value distributions

Drawdown analysis

Regime conditioned metrics

Backtesting incorporates realistic execution assumptions and estimated sweep costs to avoid optimistic bias.

No live or historical performance results are published in this repository.

# **Reproducibility and Tooling**

Python based implementation

Modular project structure

Configuration driven experiments

Git based version control and CLI workflows

Dependencies and example configuration files are included for structural reference only.

# **Access Note**

This public repository is a structural and conceptual showcase.

The full implementation, including feature logic, regime definitions, sweep cost models, and trained artifacts, is maintained in a private repository and can be shared with recruiters or interviewers upon request.
