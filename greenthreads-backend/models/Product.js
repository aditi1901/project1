// models/Product.js
const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const ProductSchema = new Schema({
    name: { type: String, required: true },
    description: { type: String, required: true },
    price: { type: Number, required: true },
    rewardPoints: { type: Number, default: 0 }
});

module.exports = mongoose.model('Product', ProductSchema);
