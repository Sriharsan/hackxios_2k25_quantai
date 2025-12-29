from src.blockchain.ethereum_anchor import EthereumAnchor

eth = EthereumAnchor()
result = eth.anchor_hash("test-hash-123")

print(result)
