"""
Seed Script — Mini-Grid & Smart Metering Knowledge Base

Populates the Supabase pgvector collection with technical documents covering
smart electricity metering, mini-grid design and deployment, solar generation,
and battery storage in the context of rural electrification in Africa.

Usage:
    python scripts/seed.py

Requires SUPABASE_DATABASE_URL and GOOGLE_API_KEY in .env
"""

import sys
import os

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

from app.config import get_settings
from app.rag import RAGService


# ============================================================
# SEED DOCUMENTS
# ============================================================

DOCUMENTS = [
    # --- Mini-grid design & sizing ---
    Document(
        page_content=(
            "Mini-grid sizing begins with a demand assessment of the target community. "
            "Typical load profiles for rural Sub-Saharan Africa show two demand peaks: "
            "a morning peak (06:00–08:00) driven by water pumping and cooking, and an "
            "evening peak (18:00–22:00) driven by lighting and phone charging. "
            "Productive use loads such as agro-processing mills and welding machines are "
            "increasingly common and must be factored into the peak demand estimate. "
            "A correctly sized mini-grid uses a load factor analysis to balance capital "
            "cost against utilisation rate, typically targeting a load factor of 0.25–0.40 "
            "for early-stage rural systems."
        ),
        metadata={"source": "mini_grid_design_guide.pdf", "topic": "sizing", "region": "Sub-Saharan Africa"},
    ),
    Document(
        page_content=(
            "Solar PV array sizing for a mini-grid requires calculating peak sun hours (PSH) "
            "at the project site. In the Sahel and East Africa, PSH ranges from 5.0 to 6.5 "
            "hours per day on average, while Central African forest zones average 4.0–4.5 PSH. "
            "The array must cover daily demand plus battery charging losses (typically 85–90% "
            "round-trip efficiency for lithium-ion). A standard sizing formula is: "
            "Array kWp = (Daily kWh demand / PSH) × 1.25 safety factor. "
            "String configuration must respect the charge controller's maximum input voltage "
            "(typically 150 Vdc for MPPT controllers) and account for temperature de-rating "
            "at ambient temperatures above 25°C."
        ),
        metadata={"source": "solar_pv_sizing_manual.pdf", "topic": "solar_sizing"},
    ),
    Document(
        page_content=(
            "Battery storage selection for rural mini-grids involves a trade-off between "
            "capital cost, cycle life, and maintenance requirements. Lead-acid (LA) batteries "
            "remain common due to low upfront cost, but require regular electrolyte topping-up "
            "and have a cycle life of 300–500 cycles at 50% depth of discharge (DoD). "
            "Lithium iron phosphate (LiFePO4) batteries offer 2,000–4,000 cycles at 80% DoD "
            "and are maintenance-free, but cost 2–3× more per kWh of usable capacity. "
            "For sites with skilled maintenance, LA is viable. For remote or operator-light "
            "deployments, LiFePO4 is preferred despite the higher CAPEX. "
            "Battery sizing should provide a minimum of 2–3 days of autonomy for critical loads."
        ),
        metadata={"source": "battery_storage_guide.pdf", "topic": "battery_storage"},
    ),

    # --- Smart metering & prepaid ---
    Document(
        page_content=(
            "Prepaid electricity metering is the dominant revenue model for mini-grids in "
            "rural Africa. The Standard Transfer Specification (STS, IEC 62055-41) defines "
            "the token format used to transfer credit between a vending system and a prepaid "
            "meter. Tokens are 20-digit numerical codes generated using DES encryption and "
            "are meter-specific to prevent fraud. Customers purchase tokens via mobile money "
            "(M-Pesa, MTN MoMo, Orange Money) or through a local agent. On entry, the meter "
            "decrypts the token and credits the corresponding kWh units. Tariffs can be "
            "tiered — for example, 0–5 kWh at a lifeline rate, then a higher rate above — "
            "configured in the meter at installation."
        ),
        metadata={"source": "prepaid_metering_handbook.pdf", "topic": "prepaid_metering"},
    ),
    Document(
        page_content=(
            "Smart meters in mini-grids communicate consumption data and alarms to a central "
            "head-end system (HES) using one of three connectivity approaches: "
            "1. GPRS/4G cellular — most common, requires SIM card with a data plan; "
            "2. LoRaWAN — low-power wide area network, suitable for dense settlements, "
            "   range up to 5 km line-of-sight, requires a gateway at the mini-grid site; "
            "3. Power Line Communication (PLC) — uses the LV distribution network as the "
            "   communication medium, avoids the need for a separate radio network. "
            "Data collected includes energy consumption (kWh), tamper events, power quality "
            "parameters (voltage, frequency, power factor), and meter health status. "
            "Remote disconnection via a relay inside the meter allows the operator to manage "
            "defaulters without a field visit."
        ),
        metadata={"source": "smart_meter_comms_guide.pdf", "topic": "smart_metering"},
    ),
    Document(
        page_content=(
            "The IEC 62056 DLMS/COSEM standard defines the data model and communication "
            "protocol used by smart meters. COSEM (Companion Specification for Energy "
            "Metering) structures meter data into objects — each representing a measurement "
            "or configuration parameter — accessed via the DLMS (Device Language Message "
            "Specification) protocol over HDLC or TCP/IP transport layers. "
            "In mini-grid deployments, DLMS/COSEM enables interoperability between meters "
            "from different manufacturers and the head-end system, allowing operators to "
            "switch vendors without replacing their entire data infrastructure. "
            "Key COSEM objects for mini-grid monitoring: Load Profile (interval consumption), "
            "Disconnect Control (remote relay), Event Log (tamper and alarm history), "
            "and Register (instantaneous measurements)."
        ),
        metadata={"source": "dlms_cosem_overview.pdf", "topic": "metering_standards"},
    ),

    # --- Pay-As-You-Go (PAYG) ---
    Document(
        page_content=(
            "Pay-As-You-Go (PAYG) mini-grid models use IoT-enabled meters with embedded "
            "credit control to allow customers to pay for electricity incrementally, often "
            "daily or weekly. Unlike traditional STS prepaid, PAYG meters connect to a cloud "
            "platform that can push credit remotely via SMS or data, without the customer "
            "entering a token code. "
            "PAYG dramatically lowers the effective connection cost barrier: instead of "
            "paying a USD 150–400 connection fee upfront, a customer may pay USD 5–10 per "
            "month and own the meter after 2–3 years. "
            "Key PAYG platform features: mobile money integration, automated credit scoring "
            "based on payment history, remote load limiting (to prevent theft), and data "
            "dashboards for portfolio performance monitoring."
        ),
        metadata={"source": "payg_minigrid_model.pdf", "topic": "payg", "region": "Sub-Saharan Africa"},
    ),

    # --- Grid protection & power quality ---
    Document(
        page_content=(
            "Mini-grid protection systems must detect and isolate faults quickly to prevent "
            "equipment damage and ensure safety. A typical LV mini-grid protection scheme "
            "includes: an over-current relay (ANSI 50/51) at the inverter output, earth "
            "fault protection (ANSI 51N) on the distribution feeders, and under/over-voltage "
            "relays (ANSI 27/59) to trip loads during power quality events. "
            "For solar-diesel hybrid systems, the anti-islanding protection (ANSI 81O/U) "
            "prevents the inverter from energising a de-energised feeder when the diesel "
            "generator is offline. "
            "Discrimination between the protection zones requires careful relay co-ordination "
            "to ensure the faulted feeder is isolated without tripping the entire mini-grid. "
            "Arc-fault circuit interrupters (AFCIs) are increasingly specified at customer "
            "connection points in thatch-roof settlements to reduce fire risk."
        ),
        metadata={"source": "minigrid_protection_guide.pdf", "topic": "grid_protection"},
    ),
    Document(
        page_content=(
            "Power quality in rural mini-grids is a common source of customer complaints and "
            "appliance damage. Key parameters to monitor and control: "
            "Voltage regulation: IEC 60038 specifies nominal LV as 230 V ±10%. Mini-grids "
            "with long feeder runs often see voltage drops of 8–12% at end-of-line customers "
            "during peak demand, requiring conductor upsizing or reactive power compensation. "
            "Frequency: Off-grid inverters regulate frequency to 50 Hz ±2%. Severe underfrequency "
            "can occur when solar output drops suddenly (cloud transients) and battery state-of-charge "
            "is low, requiring load shedding by the Energy Management System (EMS). "
            "Harmonics: Non-linear loads (phone chargers, LED drivers, variable-speed pumps) "
            "introduce current harmonics. Total Harmonic Distortion (THD) above 8% can cause "
            "transformer overheating and metering errors in older electromechanical meters."
        ),
        metadata={"source": "power_quality_minigrid.pdf", "topic": "power_quality"},
    ),

    # --- Rural electrification context ---
    Document(
        page_content=(
            "The Multi-Tier Framework (MTF), developed by the Sustainable Energy for All "
            "initiative (SEforALL) and the World Bank, defines energy access in five tiers "
            "based on capacity, duration, reliability, quality, affordability, legality, and "
            "health and safety. Tier 1 provides at least 12 Wh/day for 4 hours (basic lighting "
            "and phone charging). Tier 3 (50 W, 8 hours/day) is the minimum to support a "
            "small business or productive use. Tier 4 and 5 are equivalent to national grid "
            "standards. Mini-grids are typically designed to deliver Tier 3–4 access. "
            "The MTF is used by development finance institutions (DFIs) to set project "
            "targets and evaluate impact for programmes in Nigeria, DRC, Tanzania, Kenya, "
            "and Ethiopia — five of the largest electrification deficits in Africa."
        ),
        metadata={"source": "mtf_energy_access_framework.pdf", "topic": "energy_access", "region": "Sub-Saharan Africa"},
    ),
    Document(
        page_content=(
            "Mini-grid deployment in rural Africa faces several non-technical barriers. "
            "Land tenure and wayleave: obtaining rights-of-way for distribution lines across "
            "customary land requires negotiation with traditional authorities, which can add "
            "months to the development timeline. "
            "Regulatory uncertainty: in many countries, the electricity act requires a "
            "generation or distribution licence for any system above 100 kW, creating a "
            "compliance burden for small private developers. Some regulators (e.g. Nigeria's "
            "NERC, Tanzania's EWURA) have introduced simplified licencing for mini-grids "
            "below a threshold capacity. "
            "Grid arrival risk: if the national utility extends the grid to a mini-grid "
            "community, the developer may lose customers unless there is a buyout or "
            "interconnection framework in place. DFIs such as the AfDB and IFC increasingly "
            "require grid arrival risk mitigation clauses in their financing agreements."
        ),
        metadata={"source": "minigrid_deployment_barriers.pdf", "topic": "deployment", "region": "Sub-Saharan Africa"},
    ),
    Document(
        page_content=(
            "Operation and maintenance (O&M) of rural mini-grids presents logistical "
            "challenges due to site remoteness. Best-practice O&M frameworks distinguish "
            "between preventive maintenance (scheduled), corrective maintenance (fault "
            "response), and predictive maintenance (condition-based). "
            "Smart metering data enables predictive maintenance: voltage imbalance trends "
            "can indicate a failing connection before a fault occurs; unusual consumption "
            "patterns may signal meter tampering or a faulty appliance drawing excessive "
            "current. "
            "Spare parts management is critical. A minimum inventory for a 50 kWp mini-grid "
            "typically includes: PV bypass diodes, MPPT fuses, inverter cooling fans, LV "
            "circuit breakers, and SIM cards. "
            "Field technician training programmes, such as those offered by GOGLA and "
            "GIZ's ENDEV, develop local capacity for first-line maintenance, reducing "
            "mean time to repair (MTTR) from days to hours."
        ),
        metadata={"source": "minigrid_om_framework.pdf", "topic": "operations_maintenance"},
    ),

    # --- Business models ---
    Document(
        page_content=(
            "Three mini-grid business models dominate in Sub-Saharan Africa: "
            "1. Private utility model: a private developer owns and operates the mini-grid, "
            "   recovering costs through tariffs regulated by the national energy regulator. "
            "   Examples: PowerGen (Kenya/Nigeria), ENGIE Energy Access (multiple countries). "
            "2. Public-private partnership (PPP): government provides infrastructure subsidies "
            "   (often through rural electrification funds) while a private operator manages "
            "   commercial operations. Common in Tanzania and Sierra Leone. "
            "3. Community cooperative model: a community-owned entity owns and operates the "
            "   system, with profits reinvested locally. Requires strong governance and "
            "   technical capacity within the community. "
            "The levelised cost of electricity (LCOE) for solar mini-grids has fallen from "
            "USD 0.60–1.20/kWh in 2015 to USD 0.25–0.55/kWh in 2024 for well-designed "
            "systems, driven by falling solar PV and battery costs."
        ),
        metadata={"source": "minigrid_business_models.pdf", "topic": "business_models", "region": "Sub-Saharan Africa"},
    ),
    Document(
        page_content=(
            "Energy management systems (EMS) for solar-battery mini-grids control the power "
            "flow between PV array, battery bank, optional diesel generator, and loads. "
            "Core EMS functions: maximum power point tracking (MPPT) for the PV array; "
            "battery state-of-charge (SoC) estimation using coulomb counting combined with "
            "voltage-based correction; load shedding priority logic (shedding non-critical "
            "loads first when SoC falls below a threshold, e.g. 30%); and generator start/stop "
            "control based on SoC and load demand. "
            "Advanced EMS platforms incorporate weather forecasting to pre-charge batteries "
            "before expected cloudy periods, reducing diesel run-hours. "
            "Communication interfaces: Modbus RTU/TCP for inverter and battery BMS data; "
            "MQTT over cellular for cloud telemetry; local HMI via touchscreen for site operators."
        ),
        metadata={"source": "ems_technical_guide.pdf", "topic": "energy_management"},
    ),
    Document(
        page_content=(
            "Demand-side management (DSM) in rural mini-grids shifts flexible loads to "
            "periods of high solar generation, reducing battery cycling and improving system "
            "economics. Common DSM strategies: "
            "Time-of-use (ToU) tariffs that charge lower rates during solar hours (09:00–16:00) "
            "incentivise water pumping and agro-processing during the day. "
            "Direct load control: the EMS can remotely switch off non-critical loads (water "
            "heaters, refrigeration compressors) via smart plugs or controllable meters during "
            "peak demand periods or when battery SoC is low. "
            "Smart appliance integration: solar-optimised refrigerators with larger thermal mass "
            "can pre-cool during daylight hours and coast through the night with minimal "
            "compressor cycles, reducing peak demand by up to 40% for that load type."
        ),
        metadata={"source": "dsm_minigrid_guide.pdf", "topic": "demand_side_management"},
    ),
]


# ============================================================
# MAIN
# ============================================================

def main():
    settings = get_settings()

    if not settings.supabase_database_url:
        print("ERROR: SUPABASE_DATABASE_URL is not set in your .env file.")
        sys.exit(1)

    print("=" * 60)
    print("Mini-Grid Knowledge Base — Seed Script")
    print("=" * 60)
    print(f"\nCollection : {settings.rag_collection_name}")
    print(f"Documents  : {len(DOCUMENTS)}")
    print(f"\nConnecting to Supabase and embedding documents...")
    print("(This will make embedding API calls — expect 20–40 seconds)\n")

    rag = RAGService(
        database_url=settings.supabase_database_url,
        collection_name=settings.rag_collection_name,
        k=settings.rag_k,
    )

    ids = rag.add_documents(DOCUMENTS)

    print(f"\n{'=' * 60}")
    print(f"Done! Ingested {len(ids)} documents into '{settings.rag_collection_name}'.")
    print(f"{'=' * 60}")
    print("\nTopics covered:")

    topics = sorted({doc.metadata.get("topic", "unknown") for doc in DOCUMENTS})
    for topic in topics:
        count = sum(1 for d in DOCUMENTS if d.metadata.get("topic") == topic)
        print(f"  {topic:<30} {count} doc{'s' if count > 1 else ''}")

    print("\nThe knowledge base is ready. Start the API and try a query like:")
    print('  curl -X POST http://localhost:8000/chat \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"message": "What battery technology is best for remote mini-grids?"}\'')


if __name__ == "__main__":
    main()
