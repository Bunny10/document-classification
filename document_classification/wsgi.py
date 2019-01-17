from document_classification.application import application

if __name__ == "__main__":
    application.run(host=application.config['HOST'],
                    port=application.config['PORT'])