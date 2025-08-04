#!/usr/bin/env python3
"""
Attack Analyzer Component for SGX Demo
Analyzes and reports on memory attack results
"""

import json
import time
import argparse
from typing import Dict, Any, List

class AttackAnalyzer:
    """Analyzes security attack results and generates reports."""
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_vulnerability_level(self, attack_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall vulnerability level based on attack results."""
        
        summary = attack_results.get('summary', {})
        successful_attacks = summary.get('successful_attacks', 0)
        total_attacks = summary.get('total_attacks', 0)
        
        # Calculate vulnerability score
        if total_attacks == 0:
            vulnerability_score = 0
        else:
            vulnerability_score = successful_attacks / total_attacks
        
        # Determine risk level
        if vulnerability_score >= 0.7:
            risk_level = "CRITICAL"
            risk_color = "🔴"
        elif vulnerability_score >= 0.4:
            risk_level = "HIGH"
            risk_color = "🟠"
        elif vulnerability_score >= 0.2:
            risk_level = "MEDIUM"
            risk_color = "🟡"
        else:
            risk_level = "LOW"
            risk_color = "🟢"
        
        return {
            'vulnerability_score': vulnerability_score,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'successful_attacks': successful_attacks,
            'total_attacks': total_attacks
        }
    
    def analyze_attack_techniques(self, attack_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze individual attack techniques."""
        
        techniques = []
        
        for attack_name, attack_data in attack_results.items():
            if attack_name == 'summary':
                continue
                
            if isinstance(attack_data, dict):
                technique = {
                    'name': attack_name.replace('_', ' ').title(),
                    'type': attack_data.get('attack_type', 'Unknown'),
                    'success': attack_data.get('success', False),
                    'technique': attack_data.get('technique', 'Not specified'),
                    'threat_level': attack_data.get('threat_level', 'UNKNOWN'),
                    'blocked_by': attack_data.get('blocked_by', None)
                }
                
                # Add specific details based on attack type
                if 'extracted_regions' in attack_data:
                    technique['data_extracted'] = f"{attack_data['extracted_regions']} memory regions"
                
                if 'memory_variance' in attack_data:
                    technique['side_channel_info'] = f"Memory variance: {attack_data['memory_variance']}"
                
                techniques.append(technique)
        
        return techniques
    
    def generate_healthcare_impact_analysis(self, vulnerability_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate healthcare-specific impact analysis."""
        
        risk_level = vulnerability_analysis['risk_level']
        
        if risk_level in ['CRITICAL', 'HIGH']:
            privacy_impact = "SEVERE"
            compliance_impact = "VIOLATION"
            business_impact = "HIGH"
            
            risks = [
                "Patient health data exposed to unauthorized access",
                "Medical model algorithms vulnerable to theft",
                "HIPAA/GDPR compliance violations likely",
                "Potential for medical identity theft",
                "Healthcare provider liability exposure"
            ]
            
            mitigations = [
                "Implement hardware-based protection (SGX)",
                "Encrypt sensitive data at rest and in transit",
                "Implement strict access controls",
                "Regular security audits and monitoring",
                "Staff training on data protection"
            ]
            
        elif risk_level == 'MEDIUM':
            privacy_impact = "MODERATE"
            compliance_impact = "CONCERN"
            business_impact = "MEDIUM"
            
            risks = [
                "Limited exposure of patient data",
                "Potential model parameter extraction",
                "Compliance audit findings likely"
            ]
            
            mitigations = [
                "Enhanced monitoring and detection",
                "Software-based protections",
                "Access logging and auditing"
            ]
            
        else:  # LOW
            privacy_impact = "MINIMAL"
            compliance_impact = "ACCEPTABLE"
            business_impact = "LOW"
            
            risks = [
                "Minimal risk to patient privacy",
                "Low probability of data exposure"
            ]
            
            mitigations = [
                "Maintain current security posture",
                "Regular security assessments"
            ]
        
        return {
            'privacy_impact': privacy_impact,
            'compliance_impact': compliance_impact,
            'business_impact': business_impact,
            'specific_risks': risks,
            'recommended_mitigations': mitigations
        }
    
    def compare_protection_levels(self, vulnerable_results: Dict[str, Any], 
                                protected_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare vulnerable vs protected execution results."""
        
        vulnerable_analysis = self.analyze_vulnerability_level(vulnerable_results)
        protected_analysis = self.analyze_vulnerability_level(protected_results)
        
        improvement = {
            'attack_success_reduction': (
                vulnerable_analysis['successful_attacks'] - 
                protected_analysis['successful_attacks']
            ),
            'risk_level_improvement': (
                vulnerable_analysis['risk_level'] != protected_analysis['risk_level']
            ),
            'protection_effectiveness': (
                1.0 - (protected_analysis['vulnerability_score'] / 
                      max(vulnerable_analysis['vulnerability_score'], 0.001))
            )
        }
        
        return {
            'vulnerable': vulnerable_analysis,
            'protected': protected_analysis,
            'improvement': improvement
        }
    
    def generate_technical_report(self, analysis_data: Dict[str, Any]) -> str:
        """Generate detailed technical report."""
        
        report = []
        report.append("=" * 60)
        report.append("TECHNICAL SECURITY ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        if 'comparison' in analysis_data:
            comp = analysis_data['comparison']
            report.append("EXECUTIVE SUMMARY")
            report.append("-" * 20)
            report.append(f"Vulnerable System: {comp['vulnerable']['risk_level']} risk")
            report.append(f"Protected System: {comp['protected']['risk_level']} risk")
            report.append(f"Protection Effectiveness: {comp['improvement']['protection_effectiveness']:.1%}")
            report.append("")
        
        # Attack Analysis
        if 'attack_techniques' in analysis_data:
            report.append("ATTACK TECHNIQUE ANALYSIS")
            report.append("-" * 30)
            
            for technique in analysis_data['attack_techniques']:
                status = "SUCCESS" if technique['success'] else "BLOCKED"
                report.append(f"• {technique['name']}: {status}")
                report.append(f"  Type: {technique['type']}")
                report.append(f"  Threat Level: {technique['threat_level']}")
                if technique.get('blocked_by'):
                    report.append(f"  Blocked By: {technique['blocked_by']}")
                report.append("")
        
        # Healthcare Impact
        if 'healthcare_impact' in analysis_data:
            impact = analysis_data['healthcare_impact']
            report.append("HEALTHCARE IMPACT ANALYSIS")
            report.append("-" * 30)
            report.append(f"Privacy Impact: {impact['privacy_impact']}")
            report.append(f"Compliance Impact: {impact['compliance_impact']}")
            report.append(f"Business Impact: {impact['business_impact']}")
            report.append("")
            
            report.append("Identified Risks:")
            for risk in impact['specific_risks']:
                report.append(f"  - {risk}")
            report.append("")
            
            report.append("Recommended Mitigations:")
            for mitigation in impact['recommended_mitigations']:
                report.append(f"  - {mitigation}")
            report.append("")
        
        return "\n".join(report)
    
    def analyze_demo_results(self, vulnerable_results: Dict[str, Any], 
                           protected_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Complete analysis of demo results."""
        
        print("[+] Analyzing attack results...")
        
        # Analyze vulnerable execution
        vulnerability_analysis = self.analyze_vulnerability_level(vulnerable_results)
        attack_techniques = self.analyze_attack_techniques(vulnerable_results)
        healthcare_impact = self.generate_healthcare_impact_analysis(vulnerability_analysis)
        
        analysis = {
            'vulnerability_analysis': vulnerability_analysis,
            'attack_techniques': attack_techniques,
            'healthcare_impact': healthcare_impact
        }
        
        # If protected results available, do comparison
        if protected_results:
            comparison = self.compare_protection_levels(vulnerable_results, protected_results)
            analysis['comparison'] = comparison
        
        return analysis
    
    def print_analysis_summary(self, analysis: Dict[str, Any]):
        """Print formatted analysis summary."""
        
        vuln = analysis['vulnerability_analysis']
        impact = analysis['healthcare_impact']
        
        print(f"\n{'='*50}")
        print("SECURITY ANALYSIS SUMMARY")
        print(f"{'='*50}")
        
        print(f"\n{vuln['risk_color']} OVERALL RISK LEVEL: {vuln['risk_level']}")
        print(f"   Vulnerability Score: {vuln['vulnerability_score']:.2f}")
        print(f"   Successful Attacks: {vuln['successful_attacks']}/{vuln['total_attacks']}")
        
        print(f"\n🏥 HEALTHCARE IMPACT:")
        print(f"   Privacy Impact: {impact['privacy_impact']}")
        print(f"   Compliance Impact: {impact['compliance_impact']}")
        print(f"   Business Impact: {impact['business_impact']}")
        
        if 'comparison' in analysis:
            comp = analysis['comparison']['improvement']
            print(f"\n🛡️  PROTECTION EFFECTIVENESS:")
            print(f"   Attack Reduction: {comp['attack_success_reduction']} attacks blocked")
            print(f"   Overall Improvement: {comp['protection_effectiveness']:.1%}")

def run_test():
    """Test the attack analyzer component."""
    print("Testing Attack Analyzer Component...")
    
    # Create test data
    test_results = {
        'memory_extraction': {'success': True, 'threat_level': 'CRITICAL'},
        'page_fault_analysis': {'success': True, 'threat_level': 'MEDIUM'},
        'summary': {'successful_attacks': 2, 'total_attacks': 2}
    }
    
    analyzer = AttackAnalyzer()
    
    try:
        analysis = analyzer.analyze_demo_results(test_results)
        print("[+] Analysis completed successfully")
        print(f"[+] Risk level: {analysis['vulnerability_analysis']['risk_level']}")
        print("[+] Component test passed")
        return True
        
    except Exception as e:
        print(f"[!] Component test failed: {e}")
        return False

def main():
    """Main analyzer entry point."""
    parser = argparse.ArgumentParser(description="Attack Result Analyzer")
    parser.add_argument("--test", action="store_true",
                       help="Run component test")
    parser.add_argument("--results-file", type=str,
                       help="JSON file with attack results")
    args = parser.parse_args()
    
    if args.test:
        return 0 if run_test() else 1
    
    if args.results_file:
        try:
            with open(args.results_file, 'r') as f:
                results = json.load(f)
            
            analyzer = AttackAnalyzer()
            
            vulnerable_results = results.get('vulnerable_execution', {})
            protected_results = results.get('sgx_execution', {})
            
            analysis = analyzer.analyze_demo_results(vulnerable_results, protected_results)
            analyzer.print_analysis_summary(analysis)
            
            # Generate detailed report
            report = analyzer.generate_technical_report(analysis)
            print(f"\n{report}")
            
            return 0
            
        except Exception as e:
            print(f"[!] Analysis failed: {e}")
            return 1
    
    print("[!] No action specified. Use --test or --results-file")
    return 1

if __name__ == "__main__":
    exit(main())