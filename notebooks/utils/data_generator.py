import random
from datetime import datetime, timedelta

def generate_financial_comments(buis_list, start_date, end_date, factors, pfs, desks, currencies):
    comments = []
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    for single_date in (start_dt + timedelta(days=n) for n in range((end_dt - start_dt).days + 1)):
        date_str = single_date.strftime("%Y-%m-%d")
        
        for buis in buis_list:
            num_entries = random.randint(1, 3)  # Each day can have 1-3 records for variation
            
            for _ in range(num_entries):
                net = random.randint(50000, 2000000)  # Random net amount
                factor = random.choice(factors)
                prof_loss = random.choice(["PROFFIT", "LOSS"])
                cur = random.choice(currencies)
                pf = random.choice(pfs)
                dsk = random.choice(desks)

                comment = (
                    f"For business {buis}, on {date_str}, driven by {factor} {net} {cur} {prof_loss} "
                    f"to PL on {cur} on Portfolio {pf} at {dsk}"
                )

                comments.append({
                    "BUIS": buis,
                    "DATE": date_str,
                    "NET": str(net),
                    "FACTOR": factor,
                    "PROF_LOSS": prof_loss,
                    "CUR": cur,
                    "PF": pf,
                    "DSK": dsk,
                    "COMMENT": comment
                })

    return comments

# Example Usage
buis_list = ["CEEMAEA", "LATAM"]
start_date = "2023-06-01"
end_date = "2024-03-07"  # Shorter range for testing
factors = ["IRDelta", "FXDelta", "IRGamma", "Theta", "BondBasis", "CreditSpread"]
pfs = ["American London CEEMAEA Portfolio", "European CEEMAEA Portfolio", "LATAM Emerging Portfolio"]
desks = ["US/LDN DSK", "EU/LDN DSK", "LATAM/NYC DSK"]
currencies = ["USD", "EUR", "GBP", "CZK", "INR", "BRL", "MXN"]

generated_comments = generate_financial_comments(buis_list, start_date, end_date, factors, pfs, desks, currencies)

# Dump the generated comments to a file
import json
with open("../../sample_data/rule_based_title_comment_data.json", "w") as f:
    json.dump(generated_comments, f, indent=4)
