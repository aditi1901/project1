// controllers/productController.js
const Product = require('../models/Product');

exports.getProducts = async (req, res) => {
    try {
        const products = await Product.find();
        res.json(products);
    } catch (err) {
        console.error(err.message);
        res.status(500).send('Server error');
    }
};

exports.addProduct = async (req, res) => {
    const { name, description, price, rewardPoints } = req.body;

    try {
        const newProduct = new Product({ name, description, price, rewardPoints });
        const product = await newProduct.save();
        res.json(product);
    } catch (err) {
        console.error(err.message);
        res.status(500).send('Server error');
    }
};
