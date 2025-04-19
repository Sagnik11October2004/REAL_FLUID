# REAL_FLUID

A computational fluid dynamics (CFD) simulation that models real fluid behavior using level set in three different scenarios:

1. Dam Break
2. Drop falling in liquid
3. Fluid surface at equilibrium

## Features

This simulation implements:

- Free Surface i.e. doesnt solve for non-fluid part
- Level set method with regular redistancing
- Chorin Projection method for solving incompressible Navier-Stokes equations
- Pressure Poisson equation solver using preconditioned linear operators
- Surface tension effects
- Explicit viscosity handling
- Semi-Lagrangian advection
- Gravity as body force

## Technical Details

The simulation uses advanced numerical methods to accurately model fluid dynamics:

- **Level Set Method**: Tracks the fluid interface implicitly
- **Chorin Projection**: Enforces incompressibility constraint
- **Preconditioned Linear Operators**: Efficiently solves the pressure equation

## Installation

### Prerequisites

- C++ compiler
- Make build system

### Building the Project

Clone the repository and build using make:

```bash
git clone https://github.com/yourusername/REAL_FLUID.git
cd REAL_FLUID
make
```

The executable will be generated in the `bin` folder.

## Usage

To run the simulation:

```bash
cd bin
./levelset2D
```

## Simulation Scenarios

### 1. Dam Break

Simulates the collapse of a water column, demonstrating gravity-driven flow and wave propagation.

### 2. Drop Falling in Liquid

Models the impact of a fluid droplet on a fluid surface, highlighting surface tension effects and splash formation.

### 3. Fluid Surface at Equilibrium

Simulates a fluid surface reaching its equilibrium state, useful for studying stability and convergence.

## Output

The simulation produces visualization data that can be further processed to create animations of the fluid behavior.

## Contact

For further information you can contact me at sb23ms007@iiserkol.ac.in
