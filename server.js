const express = require('express');
const path = require('path');
const bodyParser = require('body-parser');
const knex = require('knex');
const cors = require('cors'); 
const yahooFinance = require('yahoo-finance2').default;
const db = knex({
    client: 'pg',
    connection: {
        host: '127.0.0.1',
        user: 'postgres',
        password: 'pass@123',
        database: 'Loginform'
    }
})

const app = express();

app.use(cors());

let intialPath = path.join(__dirname, "public");

app.use(bodyParser.json());
app.use(express.static(intialPath));

app.get('/', (req, res) => {
    res.sendFile(path.join("greenhome.html"));
})

app.get('/mutual', (req, res) => {
    res.sendFile(path.join("mutual.html"));
})

app.get('/sip', (req, res) =>{
    res.sendFile(path.join("index.html"));
})

app.get('/login', (req, res) => {
    res.sendFile(path.join(intialPath, "login.html"));
})

app.get('/register', (req, res) => {
    res.sendFile(path.join(intialPath, "register.html"));
})

app.post('/register-user', (req, res) => {
    const { name, email, password } = req.body;

    if(!name.length || !email.length || !password.length){
        res.json('fill all the fields');
    } else{
        db("users").insert({
            name: name,
            email: email,
            password: password
        })
        .returning(["name", "email"])
        .then(data => {
            res.json(data[0])
        })
        .catch(err => {
            if(err.detail.includes('already exists')){
                res.json('email already exists');
            }
        })
    }
})

app.post('/login-user', (req, res) => {
    const { email, password } = req.body;

    db.select('name', 'email')
    .from('users')
    .where({
        email: email,
        password: password
    })
    .then(data => {
        if(data.length){
            res.json(data[0]);
        } else{
            res.json('email or password is incorrect');
        }
    })
})

app.post('/mquotes', async (req, res) => {
    try {
        const symbols = Object.keys(req.body); // Expecting an object with stock names as keys
        
        
        if (symbols.length === 0) {
            return res.status(400).send("No symbols provided.");
        }

        const quotes = await yahooFinance.quote(symbols);
        
        if (quotes && Object.keys(quotes).length > 0) {
            res.json(quotes);
        } else {
            res.status(404).send("Market prices not found for the provided symbols.");
        }
    } catch (error) {
        console.error(error);
        res.status(500).send("An error occurred while fetching the quotes.");
    }
});

app.listen(3000, (req, res) => {
    console.log('listening on port 3000......')
})