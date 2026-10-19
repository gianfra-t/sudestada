# Sudestada

A conceptual-design framework for aircraft, with a focus on eVTOLs. Using physical models as building blocks, you state an aircraft once,
as typed data, and every model, fit, check and solver reads it from that one definition. Checks to new configurations become CI pipelines 
on the model.

**This is an opinionated ontology, a strong type system used as the domain model.** The intention is to keep this as agnostic
to specific solvers or techniques as possible.

It doubles down as documentation and material for mathematical concepts and models. 

## Philosophy

- **The types are the ontology.** An aircraft and it's models are a set of dataclasses that acts as building blocks: parts, hinges, surfaces, rotors,
  motors and the battery. Behaviour lives behind `Protocol`s, for example a rotor model is `RotorInflow -> RotorLoads`
  and an airframe model is `AirframeState -> AerodynamicWrench`. Each part is a lego brick with a typed contract.

- **A single source of truth for the aircraft.** Any aircraft modeled has enough detail using the building blocks such that these can act
  as the "digital twin" of it, at least in the conceptual sense. Balancing the level of abstraction such that enough detail is represented but not 
  too much to bloat the framework is the challenge. 

- **Shared types across models.** The mass card and the aero assembly hang on the same `Skeleton` of parts and
  joints, because they are one airframe modelled two ways. Every aero model takes its shape from the `AeroAssembly`, and every rotor model
  takes its shape from the `Rotor`.

- **Design related types: Knobs and Uncertainties.** Modeling the design space by allowing types to act non-fixed. 
  - A `Knob[T]` is a design choice, and something that can and probably will be tweaked. The most atomic element of a parametric definition
  for an aircraft.
  - An `Uncertainty(low, high, estimate)` is a value whose bounds are known but whose true value is not. It can then be used to orchestrate
  the analysis and metrics with knowledge of the estimates. For example, a corner sweep across all uncertain values for a given check.

- **First principles first.** Any high-fidelity result has to be reconcilable with a low-fidelity check. Several
  fidelities on the same geometry can be used to build a model with some uncertainty range on it's params, which means error bars rather than a single point.

- **Solver-agnostic.** Clear separation between solver <-> Model <-> Analysis or Simulation.

- **Models are simulation ready**. Physical models share the contract with a 6-dof simulator such that the same Aerodynamics model, Power model, etc can be used both for simple checks and as core of the high fidelity dynamics simulator.


## Connection and coupling to 6-DOF

This project and this repo in particular is strongly connected to an extension of `rotorpy` that uses `pydrake`, to create a rigid-multibody representation of the aircraft, for final conecptual validation and control.

