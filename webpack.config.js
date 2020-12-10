const path = require('path');
const HTMLWebpackPlugin = require("html-webpack-plugin");
const {CleanWebpackPlugin} = require("clean-webpack-plugin");
const MonacoWebpackPlugin = require("monaco-editor-webpack-plugin");

module.exports = {
  mode: "development",
  entry: path.resolve(__dirname, './src/index.ts'),
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
      {
        test: /\.css$/,
        use: ['style-loader', 'css-loader']
      }, 
      {
        test: /\.ttf$/,
        use: ['file-loader']
      },
      {
        test: /\.(woff|woff2|ttf)(\?v=\d+\.\d+\.\d+)?$/,
          use: {
            loader: 'url-loader',
            options: {
              // Limit at 50k. Above that it emits separate files
              limit: 50000,
              // Output below fonts directory
              name: './fonts/[name].[ext]',
            },
          },
      }
    ],
  },
  resolve: {
    extensions: ['.tsx', '.ts', '.js'],
  },
  output: {
    filename: 'bundle-[hash].js',
    path: path.resolve(__dirname, 'dist'),
  },
  plugins: [
    new MonacoWebpackPlugin(), 
    new CleanWebpackPlugin(),
    new HTMLWebpackPlugin({
    	template: path.resolve(__dirname, "./src/templates/index.html")
    })
  ],
  // TODO: make a production version of this config
  devtool: "inline-source-map",
  devServer: {
    contentBase: path.join(__dirname, 'dist'),
    port: 3000
  }
};
