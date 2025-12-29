from src.blockchain.ethereum_anchor import EthereumAnchor

def main():
    eth = EthereumAnchor()

    test_hash = "test-portfolio-hash-123"
    result = eth.anchor_hash(test_hash)

    print("Ethereum Network:", result["network"])
    print("TX HASH:", result["tx_hash"])
    print("Length:", len(result["tx_hash"]))
    print("Starts with 0x:", result["tx_hash"].startswith("0x"))

    print("\nEtherscan URL:")
    print(f"https://sepolia.etherscan.io/tx/{result['tx_hash']}")

if __name__ == "__main__":
    main()
