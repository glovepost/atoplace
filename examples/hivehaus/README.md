# hivehaus

Collection of avant-garde products for the smart home.

We are using [atopile](https://atopile.io) to design a reusable electronics library for building smart home products. The idea is that most smart home products have the same basic components and can be built with the same tools. 
We are using [esphome](https://esphome.io) to build the firmware for the products.
All pcbs are designed to be manufactured by [JLCPCB](https://jlcpcb.com).


## 📦 Products

- [Hydrohomie](./src/products/hydrohomie) - Scale measuring your water intake per day.

## 🚀 Getting Started

### Prerequisites

- **[atopile](https://docs.atopile.io/atopile/quickstart)** compiler installed
- KiCad 8.0 or later (for PCB viewing/editing)
- Basic SMD soldering equipment

### Building HiveHaus products

1. **Clone the repository**

   ```bash
   git clone https://github.com/atopile/hivehaus
   cd hivehaus/products/<product-name>
   ```

2. **Compile the hardware design**

   ```bash
   ato build
   ```

3. **Generate manufacturing files**

   ```bash
   ato build -t all
   ```

4. **Order PCBs and components**
   - Upload gerbers from `/build` to your preferred PCB manufacturer
   - Use the generated BOM for component ordering

---

## 🤝 Contributing

We love contributions! Whether it's:

- 🐛 Bug fixes
- ✨ New features
- 📚 Documentation improvements
- 🎨 Design enhancements

---

## 📄 License

HiveHaus is open source hardware, released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🌟 Acknowledgments

Built with ❤️ using **[atopile](https://atopile.io)** - the language that's revolutionizing hardware design.

Special thanks to the atopile community for making hardware development accessible to everyone.

---

<div align="center">
  
**Ready to build your own smart home products?**

[📖 Read the Docs](https://github.com/atopile/hivehaus) • [💬 Join Discord](https://discord.com/invite/CRe5xaDBr3) • [⭐ Star on GitHub](https://github.com/atopile/hivehaus)

</div>
