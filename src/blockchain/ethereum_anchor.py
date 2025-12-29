import os
from web3 import Web3
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

class EthereumAnchor:
    def __init__(self):
        self.rpc_url = os.getenv("ETH_RPC_URL")
        self.private_key = os.getenv("ETH_PRIVATE_KEY")

        if not self.rpc_url or not self.private_key:
            raise ValueError("Ethereum credentials missing")

        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))

        if not self.w3.is_connected():
            raise ConnectionError("Failed to connect to Ethereum RPC")

        self.account = self.w3.eth.account.from_key(self.private_key)
        
        # 🔗 PortfolioAnchor smart contract
        self.contract_address = Web3.to_checksum_address(
            os.getenv("PORTFOLIO_CONTRACT_ADDRESS")
        )

        self.contract_abi = [
            {
                "anonymous": False,
                "inputs": [
                    {
                        "indexed": True,
                        "internalType": "bytes32",
                        "name": "portfolioHash",
                        "type": "bytes32"
                    },
                    {
                        "indexed": True,
                        "internalType": "address",
                        "name": "anchoredBy",
                        "type": "address"
                    },
                    {
                        "indexed": False,
                        "internalType": "uint256",
                        "name": "timestamp",
                        "type": "uint256"
                    }
                ],
                "name": "PortfolioAnchored",
                "type": "event"
            },
            {
                "inputs": [
                    {
                        "internalType": "bytes32",
                        "name": "portfolioHash",
                        "type": "bytes32"
                    }
                ],
                "name": "anchorPortfolio",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            }
        ]

        self.contract = self.w3.eth.contract(
            address=self.contract_address,
            abi=self.contract_abi
        )


    def anchor_hash(self, data_hash: str) -> dict:
        
        """
        Anchors a portfolio hash via PortfolioAnchor smart contract
        """
        nonce = self.w3.eth.get_transaction_count(self.account.address)

        latest_block = self.w3.eth.get_block("latest")
        base_fee = latest_block["baseFeePerGas"]

        priority_fee = self.w3.to_wei(1, "gwei")
        max_fee = base_fee + priority_fee * 2

        tx = self.contract.functions.anchorPortfolio(
            Web3.to_bytes(hexstr=data_hash)
        ).build_transaction({
            "from": self.account.address,
            "nonce": nonce,
            "gas": 120000,
            "maxFeePerGas": max_fee,
            "maxPriorityFeePerGas": priority_fee,
            "chainId": self.w3.eth.chain_id
        })
    
        signed_tx = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)

        return {
            "tx_hash": "0x" + tx_hash.hex(),
            "network": "sepolia",
            "timestamp": datetime.utcnow().isoformat()
        }
