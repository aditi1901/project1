// controllers/cartController.js
const Product = require('../models/Product');
const User = require('../models/user');

exports.addToCart = async (req, res) => {
    const { productId } = req.body;

    try {
        const product = await Product.findById(productId);
        if (!product) return res.status(404).json({ msg: 'Product not found' });

        // Logic to add product to user's cart
        res.json({ msg: 'Product added to cart' });
    } catch (err) {
        console.error(err.message);
        res.status(500).send('Server error');
    }
};
