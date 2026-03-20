# Fleet scope, economizers, demand response (the “7th problem”)

## Why it’s hard

Single-node control is a **local** feedback loop. **Economizers**, **chilled water plant optimization**, and **grid demand response** need:

- Plant-level state (wet bulb, valve positions, staging)  
- **Contracts** (what you’re allowed to change and when)  
- **Coordination** so rack-level changes don’t fight the chiller  

That’s a **product boundary**: CooledAI can expose hooks and recommendations, but the **BMS / EPMS** owner usually executes plant moves.

## Pragmatic phases

1. **Single rack / row** — stable agent + API + survey + FOPDT (where we are).  
2. **Row / influence map** — `InfluenceMap` + discovery; avoid one rack starving another’s inlet.  
3. **Read-only plant tags** — BACnet/SNMP points for OA temp, economizer mode, leaving chilled water temp; **suggest** setpoint shifts, human or BMS script approves.  
4. **Closed-loop plant** — partner integration (Schneider, Trane, Johnson API) with formal fail-safe.  
5. **DR / fleet** — aggregate telemetry, pre-cool before events, coordinated setback with customer ops sign-off.

## What to tell customers now

“We optimize **what you let us command** (fans first; CRAH/chiller when exposed via BACnet/SNMP). **Fleet and grid programs** are on the roadmap and pair with your existing BMS workflows.”
