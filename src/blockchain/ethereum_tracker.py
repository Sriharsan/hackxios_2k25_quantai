# src/blockchain/ethereum_tracker.py
# Simple Ethereum integration for portfolio tracking

import json
import hashlib
from datetime import datetime
from typing import Dict, List
from unittest import result
from numpy import record
import streamlit as st
from src.blockchain.ethereum_anchor import EthereumAnchor

class EthereumPortfolioTracker:
    """
    Blockchain-ready portfolio tracking system
    Note: This is a simplified implementation for hackathon demo
    In production, would integrate with actual Ethereum smart contracts
    """
    
    def __init__(self):
        self.blockchain_records = []
        self.eth_anchor = EthereumAnchor()
        
    def create_portfolio_hash(self, portfolio_data: Dict) -> str:
        """Create cryptographic hash of portfolio state"""
        portfolio_json = json.dumps(portfolio_data, sort_keys=True)
        return hashlib.sha256(portfolio_json.encode()).hexdigest()
    
    def record_portfolio_state(
        self,
        portfolio_weights: Dict[str, float],
        portfolio_value: float,
        performance_metrics: Dict
    ) -> Dict:

        timestamp = datetime.utcnow().isoformat()

        previous_hash = (
            self.blockchain_records[-1]['hash']
            if self.blockchain_records else '0' * 64
        )

        record = {
            'block_number': len(self.blockchain_records) + 1,
            'timestamp': timestamp,
            'portfolio_weights': portfolio_weights,
            'portfolio_value': portfolio_value,
            'performance_metrics': performance_metrics,
            'previous_hash': previous_hash
        }

        record['hash'] = self.create_portfolio_hash(record)

        self.blockchain_records.append(record)
        # 🔗 Anchor hash on Ethereum Sepolia
        eth_result = self.eth_anchor.anchor_hash(record['hash'])

        record['ethereum'] = {
            'tx_hash': eth_result['tx_hash'],
            'network': eth_result['network']
        }   

        return {
            'block_number': record['block_number'],
            'hash': record['hash'],
            'timestamp': timestamp,
            'verified': True,
            'ethereum': record['ethereum']
        }

    
    def verify_chain_integrity(self) -> bool:
        """Verify blockchain integrity"""
        if not self.blockchain_records:
            return True
        
        for i in range(1, len(self.blockchain_records)):
            current = self.blockchain_records[i]
            previous = self.blockchain_records[i-1]
            
            # Verify previous hash link
            if current['previous_hash'] != previous['hash']:
                return False
            
            # Verify current hash
            current_copy = current.copy()
            stored_hash = current_copy.pop('hash')

            # 🔒 Exclude non-deterministic fields
            current_copy.pop('ethereum', None)

            computed_hash = self.create_portfolio_hash(current_copy)
            
            if stored_hash != computed_hash:
                return False
        
        return True
    
    def get_portfolio_history(self) -> List[Dict]:
        """Get complete portfolio history (immutable record)"""
        return self.blockchain_records
    
    def generate_audit_trail(self) -> str:
        """Generate comprehensive audit trail"""
        if not self.blockchain_records:
            return "No portfolio records found."
        
        audit_report = "# BLOCKCHAIN AUDIT TRAIL\n\n"
        audit_report += f"Total Blocks: {len(self.blockchain_records)}\n"
        audit_report += f"Chain Integrity: {'✅ VERIFIED' if self.verify_chain_integrity() else '❌ COMPROMISED'}\n\n"
        
        audit_report += "## Portfolio State History\n\n"
        
        for i, record in enumerate(self.blockchain_records, 1):
            audit_report += f"### Block #{i}\n"
            audit_report += f"- **Hash:** {record['hash'][:16]}...\n"
            audit_report += f"- **Timestamp:** {record['timestamp']}\n"
            audit_report += f"- **Portfolio Value:** ${record['portfolio_value']:,.2f}\n"
            audit_report += f"- **Holdings:** {len(record['portfolio_weights'])} assets\n"
            
            if 'performance_metrics' in record:
                metrics = record['performance_metrics']
                audit_report += f"- **Return:** {metrics.get('total_return', 0)*100:.2f}%\n"
                audit_report += f"- **Sharpe Ratio:** {metrics.get('sharpe_ratio', 0):.2f}\n"
            
            audit_report += "\n"
        
        return audit_report

# Streamlit integration component
def show_ethereum_tracking_tab():
    """Add this to your Streamlit app as a new tab"""
    
    st.subheader("🔗 Blockchain Portfolio Tracking")
    
    st.info("""
    **Ethereum-Anchored Audit Ledger**
    
    This feature provides blockchain-style immutable record-keeping for your portfolio:
    - Cryptographic hashing of all portfolio states
    - Immutable audit trail
    - Chain integrity verification
    - Ready for Ethereum smart contract integration
    """)
    
    # Initialize tracker
    if 'eth_tracker' not in st.session_state:
        st.session_state.eth_tracker = EthereumPortfolioTracker()
    elif not hasattr(st.session_state.eth_tracker, "blockchain_records"):
        st.session_state.eth_tracker = EthereumPortfolioTracker()
    
    tracker = st.session_state.eth_tracker
    
    # Record current portfolio state
    if st.session_state.portfolio and st.button("📝 Record Portfolio State on Chain"):
        
        # Get current analysis
        if st.session_state.analysis_data:
            analysis = st.session_state.analysis_data
            performance = analysis.get('performance_metrics', {})
        else:
            performance = {}
        
        result = tracker.record_portfolio_state(
            st.session_state.portfolio,
            st.session_state.portfolio_value,
            performance
        )

        st.success("✅ Portfolio state anchored on Ethereum")

        # Ethereum transaction hash (ON-CHAIN)
        st.write("⛓️ **Ethereum Transaction Hash**")
        st.code(result['ethereum']['tx_hash'])

        etherscan_url = f"https://sepolia.etherscan.io/tx/{result['ethereum']['tx_hash']}"
        st.markdown(f"🔍 [View on Etherscan]({etherscan_url})")

        # Portfolio hash (OFF-CHAIN)
        st.write("📄 **Portfolio State Hash (SHA-256, off-chain)**")
        st.code(result['hash'])
    
    # Show blockchain history
    if tracker.blockchain_records:
        st.subheader("📊 Blockchain History")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Blocks", len(tracker.blockchain_records))
        
        with col2:
            integrity = tracker.verify_chain_integrity()
            st.metric("Chain Integrity", "✅ Verified" if integrity else "❌ Compromised")
        
        with col3:
            latest = tracker.blockchain_records[-1]
            st.metric("Latest Block", f"#{latest['block_number']}")
        
        # Show recent blocks
        st.write("**Recent Blocks:**")
        for record in tracker.blockchain_records[-5:]:
            with st.expander(f"Block #{record['block_number']} - {record['timestamp'][:19]}"):
                st.write(f"📄 **Portfolio Hash:** `{record['hash']}`")

                if 'ethereum' in record:
                    st.write("⛓️ **Ethereum Tx:**")
                    st.code(record['ethereum']['tx_hash'])
                
                st.write(f"**Portfolio Value:** ${record['portfolio_value']:,.2f}")
                st.write("**Holdings:**")
                for symbol, weight in record['portfolio_weights'].items():
                    st.write(f"- {symbol}: {weight*100:.1f}%")
        
        # Download audit trail
        if st.button("📥 Download Audit Trail"):
            audit_trail = tracker.generate_audit_trail()
            st.download_button(
                "Download Blockchain Audit Trail",
                audit_trail,
                file_name=f"portfolio_audit_trail_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )

# Global tracker instance
ethereum_tracker = EthereumPortfolioTracker()