# Llama 3.2 in Jax 
**Why Llama 3.2?**
- According to Islam *et al.* (2025) Llama 3.2 achieves the most balanced trade-off between performance and energy efficiency. 
- Llama 3.2 is an open-weight model with a known architecture therefore allowing for weight-transfer. 

**Modifications**
- Implemented Flash Attention.
- Implemented 4bit matmul for quantisation.

**Script**
- Measuring latency and carbon intensity for all sections of the model. 

**Carbon Counting**

![[./evaluation_results/figures/component_carbon_pie.png]]
![[./evaluation_results/figures/component_carbon_pie_int4.png]]


