// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract PortfolioAnchor {
    /// @notice Emitted when a portfolio hash is anchored
    event PortfolioAnchored(
        bytes32 indexed portfolioHash,
        address indexed anchoredBy,
        uint256 timestamp
    );

    /// @notice Anchor a portfolio hash on-chain
    function anchorPortfolio(bytes32 portfolioHash) external {
        require(portfolioHash != bytes32(0), "Invalid hash");

        emit PortfolioAnchored(
            portfolioHash,
            msg.sender,
            block.timestamp
        );
    }
}
