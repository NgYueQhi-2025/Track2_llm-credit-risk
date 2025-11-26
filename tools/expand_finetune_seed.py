#!/usr/bin/env python
"""
Expand Finetune Seed Data for Gemini

Generates ~100 synthetic training examples covering:
- Negation cases (no missed payments vs missed payments)
- Intensity modifiers (rarely, sometimes, often, always)
- Temporal decay (recent vs old issues)
- Mitigating factors
- CRF structured outputs
- Mixed scenarios

Output: tools/finetune_seed_gemini.jsonl
"""

import json
import random
import os
from typing import List, Dict

# Set seed for reproducibility (configurable via environment variable)
RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))
random.seed(RANDOM_SEED)


def generate_negation_examples() -> List[Dict]:
    """Generate examples focusing on negation patterns."""
    examples = []
    
    # Positive negation patterns (low risk)
    positive_patterns = [
        ("I have no missed payments in my credit history", "low", ["no missed payments", "clean history"]),
        ("Never defaulted on any loan", "low", ["never defaulted"]),
        ("No late payments recorded", "low", ["no late payments"]),
        ("I've never had any collections", "low", ["never had collections"]),
        ("No bankruptcy in my financial history", "low", ["no bankruptcy"]),
        ("I don't have any delinquent accounts", "low", ["no delinquent accounts"]),
        ("Never missed a credit card payment", "low", ["never missed payment"]),
        ("No negative marks on credit report", "low", ["no negative marks"]),
        ("I have not defaulted on any obligation", "low", ["not defaulted"]),
        ("Never been late with rent payments", "low", ["never late"]),
        ("No history of foreclosure", "low", ["no foreclosure"]),
        ("Never had debt sent to collections", "low", ["never collections"]),
        ("I don't owe any back taxes", "low", ["no back taxes"]),
        ("No outstanding judgments", "low", ["no judgments"]),
        ("Never filed for bankruptcy protection", "low", ["never bankruptcy"]),
        ("I have not been late on auto loan", "low", ["not late", "auto loan"]),
        ("No charge-offs on record", "low", ["no charge-offs"]),
        ("Never maxed out credit cards", "low", ["never maxed out"]),
        ("I don't have any past due accounts", "low", ["no past due"]),
        ("No liens against property", "low", ["no liens"]),
    ]
    
    # Negative patterns (high risk)
    negative_patterns = [
        ("I missed payments last year", "high", ["missed payments", "last year"]),
        ("Have defaulted on a previous loan", "high", ["defaulted"]),
        ("Several late payments recently", "high", ["late payments", "recently"]),
        ("Had a collection account", "high", ["collection account"]),
        ("Filed for bankruptcy two years ago", "high", ["bankruptcy"]),
        ("Some delinquent accounts currently", "high", ["delinquent accounts"]),
        ("Missed several credit card payments", "high", ["missed payments"]),
        ("Multiple negative marks on report", "high", ["negative marks"]),
        ("Defaulted on student loan", "high", ["defaulted", "student loan"]),
        ("Late with rent multiple times", "high", ["late", "multiple times"]),
        ("Had accounts charged off", "high", ["charged off"]),
        ("Foreclosure on previous home", "high", ["foreclosure"]),
        ("Judgments filed against me", "high", ["judgments"]),
        ("Tax liens on record", "high", ["tax liens"]),
        ("Repossession of vehicle", "high", ["repossession"]),
        ("Currently 90 days past due", "high", ["90 days past due"]),
        ("Multiple collections agencies", "high", ["multiple collections"]),
        ("Maxed out all credit cards", "high", ["maxed out"]),
        ("Eviction on rental history", "high", ["eviction"]),
        ("Civil judgments unpaid", "high", ["civil judgments", "unpaid"]),
    ]
    
    # Generate examples
    for text, risk, phrases in positive_patterns:
        examples.append({
            "prompt": f"Analyze credit risk: {text}",
            "completion": {
                "summary": f"Applicant has clean credit history with {phrases[0]}.",
                "risk_level": risk,
                "risk_score": round(random.uniform(0.1, 0.3), 2),
                "risky_phrases": [],
                "mitigating_factors": phrases,
                "negation_detected": True,
                "confidence": round(random.uniform(0.8, 0.95), 2)
            }
        })
    
    for text, risk, phrases in negative_patterns:
        examples.append({
            "prompt": f"Analyze credit risk: {text}",
            "completion": {
                "summary": f"Applicant shows credit risk with {phrases[0]}.",
                "risk_level": risk,
                "risk_score": round(random.uniform(0.7, 0.9), 2),
                "risky_phrases": phrases,
                "mitigating_factors": [],
                "negation_detected": False,
                "confidence": round(random.uniform(0.75, 0.9), 2)
            }
        })
    
    return examples


def generate_intensity_examples() -> List[Dict]:
    """Generate examples with intensity modifiers."""
    examples = []
    
    intensities = [
        ("rarely misses payments", "low", 0.3, ["rarely"]),
        ("occasionally late with payments", "medium", 0.5, ["occasionally", "late"]),
        ("sometimes misses due dates", "medium", 0.6, ["sometimes misses"]),
        ("often late with bills", "high", 0.75, ["often late"]),
        ("always pays on time", "low", 0.15, []),
        ("frequently misses payments", "high", 0.8, ["frequently misses"]),
        ("never misses a payment", "low", 0.1, []),
        ("consistently late", "high", 0.85, ["consistently late"]),
        ("regularly defaults", "high", 0.9, ["regularly defaults"]),
        ("seldom has issues", "low", 0.25, []),
        ("constantly behind", "high", 0.88, ["constantly behind"]),
        ("nearly always on time", "low", 0.18, []),
        ("routinely pays late", "high", 0.82, ["routinely late"]),
        ("hardly ever late", "low", 0.22, []),
        ("perpetually in arrears", "high", 0.92, ["perpetually in arrears"]),
        ("almost never defaults", "low", 0.12, []),
    ]
    
    for phrase, risk, score, risky in intensities:
        examples.append({
            "prompt": f"Evaluate applicant who {phrase}",
            "completion": {
                "summary": f"Payment behavior shows: {phrase}",
                "risk_level": risk,
                "risk_score": score,
                "risky_phrases": risky,
                "mitigating_factors": [] if risky else ["good payment behavior"],
                "confidence": 0.85
            }
        })
    
    return examples


def generate_temporal_examples() -> List[Dict]:
    """Generate examples with temporal factors."""
    examples = []
    
    scenarios = [
        ("missed payment 5 years ago, clean since", "low", 0.35, ["old missed payment"], ["clean recent history"]),
        ("recent missed payment last month", "high", 0.8, ["recent missed payment"], []),
        ("bankruptcy 10 years ago, rebuilt credit", "medium", 0.45, ["old bankruptcy"], ["rebuilt credit"]),
        ("recent bankruptcy filing", "high", 0.95, ["recent bankruptcy"], []),
        ("late payments 3 years ago, perfect now", "low", 0.3, ["old late payments"], ["current perfect record"]),
        ("multiple recent delinquencies", "high", 0.9, ["recent delinquencies"], []),
        ("foreclosure 8 years ago, stable since", "medium", 0.5, ["old foreclosure"], ["stable since"]),
        ("just missed last payment", "high", 0.75, ["just missed payment"], []),
        ("old collection from 7 years ago", "medium", 0.4, ["old collection"], []),
        ("currently 30 days late", "high", 0.78, ["currently late"], []),
        ("charge-off from 6 years back, clean now", "low", 0.38, ["old charge-off"], ["clean now"]),
        ("recent 60 day delinquency", "high", 0.85, ["recent delinquency"], []),
        ("ancient lien removed last year", "low", 0.32, ["old lien"], ["removed"]),
        ("default last quarter", "high", 0.87, ["recent default"], []),
        ("past issues 4+ years ago", "low", 0.33, ["past issues"], ["time passed"]),
        ("very recent payment problems", "high", 0.88, ["very recent problems"], []),
    ]
    
    for text, risk, score, risky, mitigating in scenarios:
        examples.append({
            "prompt": f"Assess applicant with {text}",
            "completion": {
                "summary": f"Credit history: {text}",
                "risk_level": risk,
                "risk_score": score,
                "risky_phrases": risky,
                "mitigating_factors": mitigating,
                "temporal_factors": [text],
                "confidence": 0.82
            }
        })
    
    return examples


def generate_mitigating_examples() -> List[Dict]:
    """Generate examples with mitigating factors."""
    examples = []
    
    scenarios = [
        ("missed payment due to medical emergency, now stable with high income", "medium", 0.4),
        ("high debt but also high income and savings", "medium", 0.45),
        ("multiple loans but managing all successfully", "low", 0.35),
        ("one late payment but 10 years clean history", "low", 0.25),
        ("bankruptcy but now employed with collateral", "medium", 0.5),
        ("recent job loss but found new stable position", "medium", 0.4),
        ("high credit utilization but paying down rapidly", "medium", 0.48),
        ("missed rent once due to natural disaster, caught up", "low", 0.3),
    ]
    
    for text, risk, score in scenarios:
        # Extract mitigating factors
        mitigating = []
        if "high income" in text:
            mitigating.append("high income")
        if "savings" in text:
            mitigating.append("savings")
        if "successfully" in text:
            mitigating.append("managing well")
        if "clean history" in text:
            mitigating.append("long clean history")
        if "stable" in text or "employed" in text:
            mitigating.append("stable employment")
        if "collateral" in text:
            mitigating.append("collateral")
        if "paying down" in text:
            mitigating.append("actively reducing debt")
        
        examples.append({
            "prompt": f"Review applicant: {text}",
            "completion": {
                "summary": f"Mixed profile with mitigating factors: {text[:50]}...",
                "risk_level": risk,
                "risk_score": score,
                "risky_phrases": [],
                "mitigating_factors": mitigating,
                "confidence": 0.78
            }
        })
    
    return examples


def generate_mixed_examples() -> List[Dict]:
    """Generate complex mixed scenario examples."""
    examples = []
    
    scenarios = [
        {
            "text": "Long employment history, no missed payments, low debt-to-income ratio",
            "risk": "low",
            "score": 0.15,
            "risky": [],
            "mitigating": ["stable employment", "no missed payments", "low DTI"]
        },
        {
            "text": "Multiple recent inquiries, high utilization, sometimes late",
            "risk": "high",
            "score": 0.8,
            "risky": ["recent inquiries", "high utilization", "sometimes late"],
            "mitigating": []
        },
        {
            "text": "Never defaulted, always pays on time, diverse credit mix",
            "risk": "low",
            "score": 0.1,
            "risky": [],
            "mitigating": ["never defaulted", "always on time", "diverse credit"]
        },
        {
            "text": "Recent collections, no stable income, high debt burden",
            "risk": "high",
            "score": 0.9,
            "risky": ["recent collections", "no stable income", "high debt"],
            "mitigating": []
        },
        {
            "text": "Good credit score, occasional late payment, strong assets",
            "risk": "medium",
            "score": 0.4,
            "risky": ["occasional late payment"],
            "mitigating": ["good credit score", "strong assets"]
        },
        {
            "text": "No delinquencies, excellent payment record, sufficient income",
            "risk": "low",
            "score": 0.12,
            "risky": [],
            "mitigating": ["no delinquencies", "excellent record", "sufficient income"]
        },
        {
            "text": "Frequent late payments, multiple accounts in collections",
            "risk": "high",
            "score": 0.88,
            "risky": ["frequent late payments", "multiple collections"],
            "mitigating": []
        },
        {
            "text": "Steady employment 10 years, never missed rent, owns home",
            "risk": "low",
            "score": 0.08,
            "risky": [],
            "mitigating": ["steady employment", "never missed rent", "homeowner"]
        },
        {
            "text": "Bankruptcy last year, still rebuilding, inconsistent income",
            "risk": "high",
            "score": 0.85,
            "risky": ["bankruptcy last year", "inconsistent income"],
            "mitigating": ["rebuilding"]
        },
        {
            "text": "Low credit score but improving trend, regular payments now",
            "risk": "medium",
            "score": 0.45,
            "risky": ["low credit score"],
            "mitigating": ["improving trend", "regular payments"]
        },
    ]
    
    for scenario in scenarios:
        examples.append({
            "prompt": f"Full assessment: {scenario['text']}",
            "completion": {
                "summary": f"Comprehensive analysis: {scenario['text'][:60]}...",
                "risk_level": scenario["risk"],
                "risk_score": scenario["score"],
                "risky_phrases": scenario["risky"],
                "mitigating_factors": scenario["mitigating"],
                "confidence": 0.88
            }
        })
    
    return examples


def generate_crf_examples() -> List[Dict]:
    """Generate examples with complete CRF structure."""
    examples = []
    
    # Full CRF examples
    crf_scenarios = [
        {
            "input": "Applicant has stable job for 5 years, never missed payment, no debt",
            "output": {
                "summary": "Excellent credit profile with stable employment and perfect payment history",
                "confidence": 0.95,
                "risky_phrases": [],
                "risk_score": 0.05,
                "risk_level": "low",
                "mitigating_factors": ["stable employment", "never missed payment", "no debt"],
                "risk_factors": [],
                "negation_detected": True,
                "temporal_factors": ["5 years employment"]
            }
        },
        {
            "input": "Recently filed bankruptcy, currently unemployed, multiple collections",
            "output": {
                "summary": "Very high risk profile with recent bankruptcy and current financial instability",
                "confidence": 0.92,
                "risky_phrases": ["recently filed bankruptcy", "unemployed", "multiple collections"],
                "risk_score": 0.95,
                "risk_level": "high",
                "mitigating_factors": [],
                "risk_factors": ["recent bankruptcy", "unemployment", "collections"],
                "negation_detected": False,
                "temporal_factors": ["recent bankruptcy"]
            }
        },
        {
            "input": "No missed payments, but high credit utilization at 80%",
            "output": {
                "summary": "Good payment history but concerning credit utilization suggests medium risk",
                "confidence": 0.82,
                "risky_phrases": ["high credit utilization"],
                "risk_score": 0.55,
                "risk_level": "medium",
                "mitigating_factors": ["no missed payments"],
                "risk_factors": ["high utilization"],
                "negation_detected": True,
                "temporal_factors": []
            }
        },
        {
            "input": "I never had late payments, own property, income $75k",
            "output": {
                "summary": "Strong creditworthiness with assets and clean payment history",
                "confidence": 0.9,
                "risky_phrases": [],
                "risk_score": 0.1,
                "risk_level": "low",
                "mitigating_factors": ["never late", "property owner", "good income"],
                "risk_factors": [],
                "negation_detected": True,
                "temporal_factors": []
            }
        },
        {
            "input": "Sometimes miss payments, no savings, job for 6 months",
            "output": {
                "summary": "Moderate risk with payment issues and limited employment tenure",
                "confidence": 0.78,
                "risky_phrases": ["sometimes miss payments", "no savings"],
                "risk_score": 0.62,
                "risk_level": "medium",
                "mitigating_factors": ["employed"],
                "risk_factors": ["payment issues", "no savings", "short tenure"],
                "negation_detected": True,
                "temporal_factors": ["6 months employment"]
            }
        },
        {
            "input": "Old bankruptcy 8 years ago, never missed since, stable income",
            "output": {
                "summary": "Historical issue resolved with strong recent performance",
                "confidence": 0.85,
                "risky_phrases": [],
                "risk_score": 0.35,
                "risk_level": "low",
                "mitigating_factors": ["never missed since", "stable income", "old event"],
                "risk_factors": [],
                "negation_detected": True,
                "temporal_factors": ["8 years ago bankruptcy", "recent clean record"]
            }
        },
        {
            "input": "Defaulted last month, multiple late payments, high debt",
            "output": {
                "summary": "High risk with recent default and ongoing payment problems",
                "confidence": 0.93,
                "risky_phrases": ["defaulted last month", "multiple late payments", "high debt"],
                "risk_score": 0.89,
                "risk_level": "high",
                "mitigating_factors": [],
                "risk_factors": ["recent default", "multiple lates", "high debt"],
                "negation_detected": False,
                "temporal_factors": ["last month default"]
            }
        },
        {
            "input": "No collections, rarely late, employed 3 years, medium income",
            "output": {
                "summary": "Low risk applicant with stable profile and minimal issues",
                "confidence": 0.87,
                "risky_phrases": [],
                "risk_score": 0.25,
                "risk_level": "low",
                "mitigating_factors": ["no collections", "rarely late", "employed 3 years"],
                "risk_factors": [],
                "negation_detected": True,
                "temporal_factors": ["3 years employment"]
            }
        },
        {
            "input": "Foreclosure 2 years ago, currently renting, steady payments",
            "output": {
                "summary": "Recovering from past issue with improving payment behavior",
                "confidence": 0.75,
                "risky_phrases": ["foreclosure 2 years ago"],
                "risk_score": 0.5,
                "risk_level": "medium",
                "mitigating_factors": ["steady payments"],
                "risk_factors": ["recent foreclosure"],
                "negation_detected": False,
                "temporal_factors": ["2 years ago foreclosure", "current steady payments"]
            }
        },
        {
            "input": "Always on time, no debt, excellent credit score 800+",
            "output": {
                "summary": "Exceptional credit profile with perfect payment behavior",
                "confidence": 0.98,
                "risky_phrases": [],
                "risk_score": 0.02,
                "risk_level": "low",
                "mitigating_factors": ["always on time", "no debt", "excellent score"],
                "risk_factors": [],
                "negation_detected": True,
                "temporal_factors": []
            }
        },
    ]
    
    for scenario in crf_scenarios:
        examples.append({
            "prompt": scenario["input"],
            "completion": scenario["output"]
        })
    
    return examples


def main():
    """Generate all examples and save to JSONL."""
    print("Generating synthetic finetune examples...")
    
    all_examples = []
    
    # Generate different types of examples
    all_examples.extend(generate_negation_examples())
    print(f"Generated {len(all_examples)} negation examples")
    
    intensity = generate_intensity_examples()
    all_examples.extend(intensity)
    print(f"Added {len(intensity)} intensity examples")
    
    temporal = generate_temporal_examples()
    all_examples.extend(temporal)
    print(f"Added {len(temporal)} temporal examples")
    
    mitigating = generate_mitigating_examples()
    all_examples.extend(mitigating)
    print(f"Added {len(mitigating)} mitigating factor examples")
    
    mixed = generate_mixed_examples()
    all_examples.extend(mixed)
    print(f"Added {len(mixed)} mixed scenario examples")
    
    crf = generate_crf_examples()
    all_examples.extend(crf)
    print(f"Added {len(crf)} CRF examples")
    
    # Shuffle to mix example types
    random.shuffle(all_examples)
    
    # Save to JSONL
    output_path = "tools/finetune_seed_gemini.jsonl"
    print(f"\nSaving {len(all_examples)} examples to {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"✓ Successfully generated {len(all_examples)} examples")
    print(f"  - Negation examples: emphasis on 'no/never' patterns")
    print(f"  - Intensity modifiers: rarely/sometimes/often/always")
    print(f"  - Temporal factors: recent vs historical events")
    print(f"  - Mitigating factors: offsetting risk indicators")
    print(f"  - Mixed scenarios: complex real-world cases")
    print(f"  - CRF structured: complete schema examples")


if __name__ == '__main__':
    main()
