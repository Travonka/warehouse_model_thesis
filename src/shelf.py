from src.product import Product

class Shelf:
    def __init__(self, max_volume_liters):
        self.max_volume_liters = max_volume_liters
        self.products = []

    def add_product(self, product):
        if self.check_fit(product):
            self.products.append(product)

    def remove_product(self, product_to_remove):
        to_rem = [product for product in self.products if (product_to_remove.id == product.id and product_to_remove.exemplar_id == product.exemplar_id)]
        self.products.remove(to_rem[0])

    def check_fit(self, product):
        total_volume = sum([p.volume_liters for p in self.products]) + product.volume_liters
        return total_volume <= self.max_volume_liters

    def occupancy_ratio(self):
        total_volume = sum([p.volume_liters for p in self.products])
        return total_volume / self.max_volume_liters