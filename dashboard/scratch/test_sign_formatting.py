avg_baseline_return = 45879.30
avg_optimized_return = 45820.17
avg_uplift = avg_optimized_return - avg_baseline_return
avg_lift_pct = (avg_uplift / avg_baseline_return * 100)

if avg_uplift >= 0:
    uplift_str = f"+${avg_uplift:,.2f}"
    lift_pct_str = f"+{avg_lift_pct:.2f}%"
    card_color = "#22C55E"
else:
    uplift_str = f"-${abs(avg_uplift):,.2f}"
    lift_pct_str = f"{avg_lift_pct:.2f}%"
    card_color = "#EF4444"

val1 = f"${avg_baseline_return:,.2f}"
val2 = f"${avg_optimized_return:,.2f} ({uplift_str})"
val3 = lift_pct_str

print("Baseline:", val1)
print("Optimized:", val2)
print("Lift Pct:", val3)
print("Color:", card_color)

assert "+$-" not in val2
assert "+-" not in val3
print("TEST PASSED!")
